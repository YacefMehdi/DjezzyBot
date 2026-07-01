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
    Validation flow (now):    run, write *.generated.json, then score_catalog.py diffs it
                              against the hand-verified GOLD.

TWO BACKENDS (LLM_BACKEND env)
    "local" (DEFAULT): the SAME Qwen-7B the bot already loads, in-process — NO key, no
        network. First validation: does the extraction method work with the shipped model?
        Run it from the notebook kernel that has the bot loaded so it reuses that one model.
    "api": any OpenAI-compatible /chat/completions endpoint (Groq, Together, OpenRouter,
        Ollama/vLLM) — for the LATER test on a BIGGER open Qwen. Still open-source only:
            LLM_BACKEND=api  LLM_API_BASE=https://api.groq.com/openai/v1
            LLM_API_KEY=...  LLM_MODEL=qwen-2.5-32b   (bigger open Qwen, same family)

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

# --- Backend selection -----------------------------------------------------
# "local" (default): use the SAME Qwen-7B the bot already loads, in-process on Colab —
#   no API key, no network. This is the first validation: does the extraction METHOD work
#   with the model we actually ship? Run it in the notebook kernel that has the bot loaded
#   (import build_catalog; build_catalog.main()) so it reuses the one model — NOT as a
#   subprocess, which would load a second 7B and OOM the T4.
# "api": talk to any OpenAI-compatible endpoint (Groq/Together/…), for the LATER test on a
#   BIGGER open Qwen. Set LLM_BACKEND=api + LLM_API_KEY + LLM_MODEL.
BACKEND = os.environ.get("LLM_BACKEND", "local").lower()

# --- API config (only used when BACKEND == "api") --------------------------
API_BASE = os.environ.get("LLM_API_BASE", "https://api.groq.com/openai/v1").rstrip("/")
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "qwen-2.5-32b")

# Sane bounds: a Djezzy subscription price is between these (DA). Anything outside is
# a parsing artifact (crédit amount, phone price, typo) and is rejected.
PRICE_MIN, PRICE_MAX = 20, 60000

# How much of a page's text to feed the extractor. Kept deliberately TIGHT: an offer's real
# tiers sit at the top of its page, and feeding more (we tried 12000) pulled in OTHER sections
# --- comparison blocks, related-offer amounts --- that the model then mistook for this offer's
# tiers, adding phantom prices (Confort/Zid gained 2000/4000/6000). Less context = less noise.
PAGE_CHARS = 6000

# A genuine tier always DESCRIBES its content (data volume / calls / validity). A bare price
# with no described plan is noise — an activation/recharge/crédit amount or a promo line. We
# reject such "tiers" even if the model emits them, which is model-independent insurance
# against the phantom-price failure (Legend Pro 100/150, Campuce 300/2000/3000, iZZY 800/1000).
_TIER_CONTENT_RE = re.compile(
    r"\d+\s*(?:go|mo|gb|mb)|appel|\bmin\b|sms|illimit|jour|semaine|mois|validit|heure|24h|/h",
    re.IGNORECASE,
)


# ===========================================================================
# LLM call — local (in-process Qwen-7B) or api (OpenAI-compatible), same JSON contract
# ===========================================================================
def _parse_json(text: str) -> dict:
    """Pull the JSON object out of a model reply (strips ```json fences / stray prose)."""
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)   # outermost {...} even if wrapped in a sentence
    return json.loads(m.group(0) if m else text)


def _llm_local(system: str, user: str, max_new_tokens: int = 1024) -> str:
    """Generate with the bot's already-loaded Qwen-7B (same model, tokenizer, decode guard).

    Reuses bot.load_llm()'s cached model, so calling this from the notebook kernel that ran
    the bot does NOT spin up a second 7B. Applies Qwen's chat template exactly like the bot.
    """
    import bot
    _, tok = bot.load_llm()
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True,
    )
    return bot._generate_n(prompt, max_new_tokens)


def _llm_api(system: str, user: str, retries: int = 3) -> str:
    """POST a chat completion to the OpenAI-compatible endpoint; return the raw text."""
    if not API_KEY:
        raise RuntimeError("LLM_API_KEY is not set — export it before running (BACKEND=api).")
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
        # A browser User-Agent is REQUIRED: Groq is behind Cloudflare, which blocks the
        # default "Python-urllib" client with HTTP 403 (Cloudflare error 1010).
        headers={"Authorization": f"Bearer {API_KEY}",
                 "Content-Type": "application/json",
                 "User-Agent": config.USER_AGENT},
    )
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except (urllib.error.URLError, KeyError, ValueError) as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"API call failed after {retries} tries: {last}")


def llm_json(system: str, user: str) -> dict:
    """Return the parsed JSON object the model emits, via whichever backend is selected.

    Local Qwen-7B is greedy/deterministic, so a parse failure won't change on retry —
    one shot, then the caller keep-last-good's. The api path already retries transport.
    """
    text = _llm_local(system, user) if BACKEND == "local" else _llm_api(system, user)
    return _parse_json(text)


# ===========================================================================
# Prompts
# ===========================================================================
_OFFER_SYSTEM = (
    "Tu es un extracteur de données pour les offres télécom Djezzy. On te donne le TEXTE "
    "BRUT d'une page d'offre (souvent mal structuré). Tu renvoies UNIQUEMENT un objet JSON "
    "valide, sans texte autour, au format :\n"
    '{ "tiers": [ {"price": <entier DA>, "details": "<volume Go, appels, SMS, validité>"} ] }\n'
    "\n"
    "DÉFINITION D'UN PALIER (tier) — un FORFAIT réel qu'on achète :\n"
    "- N'émets un palier QUE si la page DÉCRIT son contenu : au minimum un volume de données "
    "(Go/Mo) OU des appels/minutes, avec sa validité. Le couple prix + contenu doit être "
    "explicitement présent dans le texte.\n"
    "- Un PRIX SEUL, sans contenu de forfait décrit à côté, n'est PAS un palier : ignore-le "
    "(c'est un montant de recharge, d'activation, une promo ou un crédit). Exemple : « rechargez "
    "dès 100 DA » n'est PAS un palier ; « 100 DA : 1 Go + appels illimités, validité 24h » en "
    "est un.\n"
    "\n"
    "EXHAUSTIVITÉ — n'oublie aucun palier :\n"
    "- Extrais TOUS les paliers de la page, du MOINS cher au PLUS cher. Ne t'arrête pas après "
    "les premiers ; n'oublie ni le moins cher ni le plus cher. Une gamme a souvent plusieurs "
    "paliers (parfois 2, parfois 8).\n"
    "- Si DEUX paliers ont le MÊME prix mais un contenu DIFFÉRENT (ex. deux forfaits à 2000 DA, "
    "l'un 90 Go l'autre 70 Go), garde-les TOUS LES DEUX comme deux entrées distinctes — ne les "
    "fusionne JAMAIS en un seul.\n"
    "\n"
    "À NE JAMAIS COMPTER COMME PRIX D'UN PALIER :\n"
    "- les montants de crédit/bonus offerts ('+2000 DA de crédit', 'bonus 500 DA') → ils vont "
    "DANS le champ details du palier concerné, jamais comme un palier séparé ;\n"
    "- les tarifs à l'unité ('5 DA/SMS', '4,99 DA/Mo', 'appel à 4 DA/min') ;\n"
    "- les frais de SIM, de transfert ou d'activation ;\n"
    "- les valeurs d'appels exprimées en DA ('4000 DA vers le national').\n"
    "\n"
    "N'INVENTE RIEN : ne déduis aucun prix, ne complète pas une gamme par des paliers supposés, "
    "ne recopie pas un exemple. Si un champ est absent, ne le mets pas. Si aucun forfait décrit "
    'n\'est présent, renvoie {"tiers": []}.\n'
    "\n"
    "Exemple de FORMAT (Djezzy Legend — montre un petit palier réel ET deux paliers au même "
    "prix) -> "
    '{"tiers":[{"price":100,"details":"1 Go internet, appels illimités vers Djezzy, '
    '+300 DA de crédit, validité 24h"},{"price":2000,"details":"90 Go internet, appels '
    'illimités, validité 30 jours"},{"price":2000,"details":"70 Go internet + 350 min hors '
    'réseau, validité 30 jours"}]}'
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
            and isinstance(t.get("details"), str) and t["details"].strip()
            # a real tier describes its content; a bare price is noise -> reject it
            and _TIER_CONTENT_RE.search(t["details"]) is not None)


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
            data = llm_json(_OFFER_SYSTEM, page["content"][:PAGE_CHARS])
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
            data = llm_json(_ROAMING_SYSTEM, page["content"][:PAGE_CHARS])
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
    if BACKEND == "local":
        print(f"Backend: local in-process model ({config.LLM_MODEL_ID})   (apply={apply})")
    else:
        print(f"Backend: api  Model: {MODEL} via {API_BASE}   (apply={apply})")
    _backup_gold()                       # snapshot gold first — a run can never lose it
    pages_by_url = _index_pages()
    t0 = time.time()                     # wall-clock of the actual extraction work

    print("\n=== OFFERS ===")
    offers, ostat = build_offers(pages_by_url)
    _write(offers, config.OFFERS_JSON, apply, ".generated.json")

    print("\n=== ROAMING ===")
    roaming, rstat = build_roaming(pages_by_url)
    _write(roaming, config.ROAMING_JSON, apply, ".generated.json")

    elapsed = time.time() - t0
    print(f"\nOffers : {ostat}")
    print(f"Roaming: {rstat}")
    print(f"Extraction time: {elapsed:.0f}s ({elapsed/60:.1f} min) "
          f"for {ostat['rebuilt'] + rstat['rebuilt']} rebuilt / "
          f"{ostat['kept'] + rstat['kept']} kept records")
    print("Done." if apply else "Draft written. Run score_catalog.py to compare vs gold.")


if __name__ == "__main__":
    main()
