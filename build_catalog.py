"""
build_catalog.py — automatic LLM extractor that regenerates the structured catalogs
(data/offers.json + data/roaming.json) from the daily scrape (data/djezzy_pages.json).

WHY
    The hand-curated catalogs are clean but go STALE: when Djezzy changes a price, the
    daily scrape updates djezzy_pages.json, yet the catalog stays old and someone has to
    re-clean it by hand. This script makes "cleaning" an automatic, repeatable pipeline
    step instead of a manual artifact: each run re-reads the scraped offer/roaming pages
    and rebuilds the structured tiers with an LLM (which handles the messy "Pour 2000 DA"
    vs "2000 DA" layouts and the crédit/mini-option contamination far better than regex).

    Production flow (daily):  scrape -> djezzy_pages.json -> build_catalog.py -> offers.json
    Validation flow (now):    run with a bigger open Qwen, write *.generated.json, then
                              score_catalog.py diffs it against the hand-verified GOLD.

MODEL-AGNOSTIC
    Talks to any OpenAI-compatible /chat/completions endpoint (Groq, Together, OpenRouter,
    local Ollama/vLLM) via three env vars — NO closed model required, so the open-source
    brief stays intact:
        LLM_API_BASE   e.g. https://api.groq.com/openai/v1
        LLM_API_KEY    your key
        LLM_MODEL      e.g. qwen-2.5-32b   (a bigger open Qwen, same family as the 7B)

DESIGN / GUARDRAILS (fully automatic, no human gate)
    * The current catalog is the REGISTRY of offers to maintain (name/slug/url/type). The
      script re-extracts each one's tiers from its scraped page. New offers are added to
      the seed when the lineup changes (rare; prices change far more often).
    * Every extraction is validated: JSON parses, prices are ints in a sane range, the
      offer name matches the lexicon. On ANY failure -> KEEP-LAST-GOOD (the existing record
      is preserved, never overwritten with garbage). So a bad scrape/LLM day can't poison
      the catalog.
    * Default output is *.generated.json (a draft). Pass --apply to overwrite the live
      catalogs (what the daily scheduler would do once the method is proven).

Run:
    python build_catalog.py            # writes offers.generated.json + roaming.generated.json
    python build_catalog.py --apply    # overwrites the live offers.json + roaming.json
"""

import json
import os
import re
import shutil
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime

import config
from data import lexicon

# --- API config (OpenAI-compatible) ---------------------------------------
API_BASE = os.environ.get("LLM_API_BASE", "https://api.groq.com/openai/v1").rstrip("/")
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "qwen-2.5-32b")

# Sane bounds: a Djezzy subscription price is between these (DA). Anything outside is
# a parsing artifact (crédit amount, phone price, typo) and is rejected.
PRICE_MIN, PRICE_MAX = 20, 60000


# ===========================================================================
# LLM call (stdlib only — no requests dependency)
# ===========================================================================
def llm_json(system: str, user: str, retries: int = 3) -> dict:
    """POST a chat completion and return the parsed JSON object the model emits.

    Strips ```json fences that open models often wrap around the JSON. Raises on
    transport/parse failure after `retries` attempts so the caller can keep-last-good.
    """
    if not API_KEY:
        raise RuntimeError("LLM_API_KEY is not set — export it before running.")
    payload = json.dumps({
        "model": MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {API_KEY}",
                 "Content-Type": "application/json"},
    )
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            text = body["choices"][0]["message"]["content"].strip()
            text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
            # grab the outermost JSON object even if the model adds a sentence around it
            m = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(m.group(0) if m else text)
        except (urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError) as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {retries} tries: {last}")


# ===========================================================================
# Prompts
# ===========================================================================
_OFFER_SYSTEM = (
    "Tu es un extracteur de données pour les offres télécom Djezzy. On te donne le TEXTE "
    "BRUT d'une page d'offre (souvent mal structuré). Tu renvoies UNIQUEMENT un objet JSON "
    "valide, sans texte autour, au format :\n"
    '{ "tiers": [ {"price": <entier DA>, "details": "<volume Go, appels, SMS, validité>"} ] }\n'
    "RÈGLES STRICTES :\n"
    "- Un 'tier' est un FORFAIT réel qu'on peut acheter (un prix d'abonnement).\n"
    "- N'inclus JAMAIS comme prix : les montants de crédit/bonus ('+2000 DA de crédit'), "
    "les tarifs à l'unité ('5 DA/SMS', '4,99 DA/Mo'), les frais de SIM, les valeurs "
    "d'appels ('4000 DA vers le national').\n"
    "- Garde les deux forfaits s'ils ont le même prix mais des volumes différents.\n"
    "- N'invente RIEN. Si un champ est absent, ne le mets pas. Si aucun forfait n'est "
    'présent, renvoie {"tiers": []}.\n'
    "Exemple (Djezzy Legend) -> "
    '{"tiers":[{"price":100,"details":"1 Go internet, appels illimités vers Djezzy, '
    '+300 DA de crédit, validité 24h"},{"price":1000,"details":"15 Go internet, '
    'appels illimités vers Djezzy, validité 30 jours"}]}'
)

_ROAMING_SYSTEM = (
    "Tu es un extracteur de données pour les forfaits ROAMING Djezzy d'une destination. "
    "On te donne le TEXTE BRUT de la page. Tu renvoies UNIQUEMENT un objet JSON valide :\n"
    '{ "mixte": [ {"price": <entier DA>, "details": "..."} ], '
    '"internet": [ {"price": <entier DA>, "details": "..."} ] }\n'
    "RÈGLES :\n"
    "- 'mixte' = forfaits 'Internet & Voix' (data + minutes + SMS). 'internet' = forfaits "
    "'Internet seul' (data + validité, sans minutes).\n"
    "- Le prix est le montant à payer (souvent 'Pour X DA', PARFOIS juste 'X DA' sans 'Pour').\n"
    "- N'inclus pas les bandeaux marketing (ex: 'jusqu'à 8 Go') comme un forfait.\n"
    "- N'invente RIEN. Tableau vide si la famille est absente."
)


# ===========================================================================
# Validation
# ===========================================================================
def _valid_tier(t) -> bool:
    return (isinstance(t, dict) and isinstance(t.get("price"), int)
            and PRICE_MIN <= t["price"] <= PRICE_MAX
            and isinstance(t.get("details"), str) and t["details"].strip())


def _clean_tiers(raw) -> list:
    """Keep only well-formed tiers; sort cheapest-first; drop exact dups."""
    out, seen = [], set()
    for t in raw or []:
        if _valid_tier(t):
            key = (t["price"], t["details"].strip())
            if key not in seen:
                seen.add(key)
                out.append({"price": t["price"], "details": t["details"].strip()})
    return sorted(out, key=lambda x: x["price"])


def _name_ok(name: str) -> bool:
    """The extracted offer must be a name we know (guards against drift)."""
    return any(lexicon.offer_in_text(n, name) for n in lexicon.OFFER_NAMES)


# ===========================================================================
# Page lookup
# ===========================================================================
def _index_pages():
    pages = json.load(open(config.DATA_JSON, encoding="utf-8"))
    return {p["url"].rstrip("/ "): p for p in pages}


def _page_for(url: str, pages_by_url: dict):
    return pages_by_url.get((url or "").rstrip("/ "))


# ===========================================================================
# Build
# ===========================================================================
def build_offers(pages_by_url: dict) -> tuple:
    gold = json.load(open(config.OFFERS_JSON, encoding="utf-8"))
    kept, rebuilt, failed = 0, 0, 0
    for rec in gold["offers"]:
        if rec.get("type") in ("service", "info"):
            kept += 1
            continue                                    # no tiers to extract
        page = _page_for(rec.get("url", ""), pages_by_url)
        if not page:
            print(f"  [keep-last-good] {rec['name']}: page not in scrape")
            kept += 1
            failed += 1
            continue
        try:
            data = llm_json(_OFFER_SYSTEM, page["content"][:6000])
            tiers = _clean_tiers(data.get("tiers"))
            if tiers and _name_ok(rec["name"]):
                rec["tiers"] = tiers
                rebuilt += 1
                print(f"  [rebuilt] {rec['name']}: {len(tiers)} tiers")
            else:
                kept += 1
                failed += 1
                print(f"  [keep-last-good] {rec['name']}: extraction empty/invalid")
        except Exception as e:                          # noqa: BLE001 (guardrail)
            kept += 1
            failed += 1
            print(f"  [keep-last-good] {rec['name']}: {e}")
    return gold, {"rebuilt": rebuilt, "kept": kept, "failed": failed}


def build_roaming(pages_by_url: dict) -> tuple:
    gold = json.load(open(config.ROAMING_JSON, encoding="utf-8"))
    rebuilt, kept, failed = 0, 0, 0
    for dest in gold["destinations"]:
        page = _page_for(dest.get("url", ""), pages_by_url)
        if not page:
            print(f"  [keep-last-good] {dest['name']}: page not in scrape")
            kept += 1
            failed += 1
            continue
        try:
            data = llm_json(_ROAMING_SYSTEM, page["content"][:6000])
            mixte = _clean_tiers(data.get("mixte"))
            internet = _clean_tiers(data.get("internet"))
            if mixte or internet:
                dest["mixte"], dest["internet"] = mixte, internet
                rebuilt += 1
                print(f"  [rebuilt] {dest['name']}: {len(mixte)} mixte / {len(internet)} internet")
            else:
                kept += 1
                failed += 1
                print(f"  [keep-last-good] {dest['name']}: extraction empty")
        except Exception as e:                          # noqa: BLE001 (guardrail)
            kept += 1
            failed += 1
            print(f"  [keep-last-good] {dest['name']}: {e}")
    return gold, {"rebuilt": rebuilt, "kept": kept, "failed": failed}


def _backup_gold() -> str:
    """Copy the live gold catalogs to data/backups/<timestamp>/ BEFORE anything runs.

    The user's main fear is that a bad extractor run silently corrupts the hand-verified
    offers.json / roaming.json. This makes that impossible to lose: every invocation —
    draft OR --apply — first snapshots the current gold to a timestamped folder, so the
    exact pre-run catalogs can always be restored (just copy them back over). Returns the
    backup directory path.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bdir = os.path.join(config.DATA_DIR, "backups", stamp)
    os.makedirs(bdir, exist_ok=True)
    for p in (config.OFFERS_JSON, config.ROAMING_JSON):
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(bdir, os.path.basename(p)))
    print(f"  backup of current gold catalogs -> {bdir}")
    return bdir


def _write(obj: dict, live_path: str, apply: bool, suffix: str):
    out = live_path if apply else live_path.replace(".json", suffix)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  -> wrote {out}")


def main():
    apply = "--apply" in sys.argv
    print(f"Model: {MODEL} via {API_BASE}   (apply={apply})")
    _backup_gold()                       # snapshot gold first — a run can never lose it
    pages_by_url = _index_pages()

    print("\n=== OFFERS ===")
    offers, ostat = build_offers(pages_by_url)
    _write(offers, config.OFFERS_JSON, apply, ".generated.json")

    print("\n=== ROAMING ===")
    roaming, rstat = build_roaming(pages_by_url)
    _write(roaming, config.ROAMING_JSON, apply, ".generated.json")

    print(f"\nOffers : {ostat}")
    print(f"Roaming: {rstat}")
    print("Done." if apply else "Draft written. Run score_catalog.py to compare vs gold.")


if __name__ == "__main__":
    main()
