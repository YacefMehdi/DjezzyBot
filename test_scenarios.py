"""
test_scenarios.py — 14 acceptance scenarios + latency capture.

Runs the bot end-to-end against the live FAISS index and checks the behaviours
the spec requires. Each scenario is tagged with a language (ar/fr/en/dz) and a
route (catalogue/budget/roaming/named-offer/competitor/out-of-domain/context/
language) so REPORT.md can break accuracy down by both.

Because every call goes through bot.answer()/voice.voice_answer(), the latency
wrapper in bot.py records real timings into bot.LATENCY as a side effect — so by
the time this script finishes, the numbers REPORT.md needs are already collected.

Usage
-----
    python test_scenarios.py            # boots the index, runs all, prints summary
    from test_scenarios import run_all  # returns the results list for the report

Each result: {id, name, lang, route, passed, note}.
"""

import re
import sys
import logging

import config
import bot
import retriever
from retriever import smart_retrieve, _chunk_price

logger = logging.getLogger("djezzybot.tests")

_PRICE_RE = retriever._PRICE_RE
_ARABIC_RE = re.compile(r"[؀-ۿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")


# ===========================================================================
# small assertion helpers
# ===========================================================================
def _prices(text: str):
    out = []
    for m in _PRICE_RE.finditer(text.lower()):
        try:
            out.append(int(m.group(1).replace(" ", "")))
        except ValueError:
            pass
    return out


_CREDIT_WORDS_T = ("credit", "crédit", "crèdit", "recharge", "bonus", "cadeau", "offert")


def _offer_prices(text: str):
    """DA amounts in `text` that are offer PRICES (excluding crédit/bonus amounts).

    A budget answer may legitimately mention "400 DA avec 2000 DA de crédit"; the
    2000 is included value, not a price, so it must not count against the budget.
    """
    t = text.lower()
    out = []
    for m in _PRICE_RE.finditer(t):
        window = t[max(0, m.start() - 25): m.end() + 25]
        if any(w in window for w in _CREDIT_WORDS_T):
            continue
        try:
            out.append(int(m.group(1).replace(" ", "")))
        except ValueError:
            pass
    return out


def _count_offer_mentions(text: str) -> int:
    from data.lexicon import OFFER_NAMES
    t = text.lower()
    return sum(1 for n in set(OFFER_NAMES) if n in t)


def _is_mostly_arabic(text: str) -> bool:
    """Arabic by SCRIPT RATIO, not keywords — robust to short replies.

    Offer/destination names stay in Latin (Legend, iZZY...), so we don't require
    100% Arabic; a clear majority of Arabic letters is enough.
    """
    ar = len(_ARABIC_RE.findall(text))
    lat = len(_LATIN_RE.findall(text))
    return ar > 0 and ar / (ar + lat) >= 0.4


def _is_english(text: str) -> bool:
    """English via langdetect (robust to short correct replies like
    'iZZY costs 1000 DA monthly.'). Never English if it contains Arabic script.
    Falls back to a light marker check only if langdetect is unavailable/unsure.
    """
    if _ARABIC_RE.search(text):
        return False
    t = f" {text.lower()} "
    # Veto: clear French function words → not English (guards langdetect misfires
    # on short brand-heavy strings). English replies won't contain these.
    french = (" pour ", " offre ", " avec ", " vous ", " les ", " des ", " votre ",
              " est ", " gratuit ", " forfait ", " jours ", " mois ", " et ")
    if any(w in t for w in french):
        return False
    try:
        from langdetect import detect
        if detect(text) == "en":
            return True
    except Exception:
        pass
    # fallback: a common English word is present
    markers = (" the ", " is ", " are ", " for ", " you ", " with ", " costs ",
               " monthly ", " per ", " offers ", " plan ", " gives ", " includes ")
    return sum(m in t for m in markers) >= 1


# ===========================================================================
# scenarios — each returns (passed: bool, note: str)
# ===========================================================================
def t01_catalogue(idx):
    """'vos offres?' lists >= 4 offers with prices."""
    r = bot.answer("Quelles sont vos offres ?", idx)
    n_offers = _count_offer_mentions(r["text"])
    n_prices = len(_prices(r["text"]))
    ok = n_offers >= 4 and n_prices >= 4
    return ok, f"offers={n_offers} prices={n_prices}"


def t02_budget(idx):
    """'j'ai 500 DA' shows ONLY offers <= 500 DA (checked in the ANSWER text)."""
    budget = 500
    # routing-level guarantee: every kept chunk's cheapest tier is <= budget
    docs = smart_retrieve("j'ai 500 DA", "fr", idx)
    chunk_prices = [p for p in (_chunk_price(d.page_content) for d in docs) if p is not None]
    routing_ok = len(chunk_prices) > 0 and all(p <= budget for p in chunk_prices)
    # answer-level: no OFFER PRICE above budget leaks into the reply (crédit/bonus
    # amounts, which may exceed budget, are excluded from the check).
    r = bot.answer("j'ai 500 DA, qu'est-ce que vous proposez ?", idx)
    op = _offer_prices(r["text"])
    answer_ok = len(op) > 0 and all(p <= budget for p in op)
    ok = routing_ok and answer_ok
    return ok, f"answer_offer_prices={op} chunk_prices={chunk_prices}"


def t03_named_campuce(idx):
    """'parle-moi de Campuce' returns Campuce, no mixing with other gammes."""
    r = bot.answer("Parle-moi de l'offre Campuce", idx)
    t = r["text"].lower()
    has_campuce = "campuce" in t or "cam puce" in t
    others = [n for n in ("legend", "izzy", "zid", "confort") if n in t]
    ok = has_campuce and not others
    return ok, f"campuce={has_campuce} other_gammes={others}"


def t04_comparison(idx):
    """'différence entre Legend et iZZY' shows both with prices."""
    r = bot.answer("Quelle est la différence entre Legend et iZZY ?", idx)
    t = r["text"].lower()
    ok = "legend" in t and "izzy" in t and len(_prices(r["text"])) >= 2
    return ok, f"legend={'legend' in t} izzy={'izzy' in t} prices={_prices(r['text'])}"


def t05_roaming(idx):
    """Roaming France returns roaming context, not the national tariff."""
    docs = smart_retrieve("je voyage en France, quel roaming ?", "fr", idx)
    if docs == config.COMPETITOR_SENTINEL:
        return False, "unexpected competitor route"
    blob = " ".join(d.page_content.lower() + " " +
                    d.metadata.get("source_url", "").lower() for d in docs)
    ok = "roaming" in blob or "france" in blob
    return ok, f"roaming_ctx={'roaming' in blob} n_docs={len(docs)}"


def t06_english(idx):
    """English query -> English reply."""
    r = bot.answer("What internet offers do you have?", idx)
    ok = r["lang"] == "en" and _is_english(r["text"])
    return ok, f"lang={r['lang']}"


def t07_arabic(idx):
    """Arabic query -> Arabic reply."""
    r = bot.answer("ما هي عروض الإنترنت لديكم؟", idx)
    ok = r["lang"] == "ar" and _is_mostly_arabic(r["text"])
    return ok, f"lang={r['lang']} arabic={_is_mostly_arabic(r['text'])}"


def t08_darija(idx):
    """Darija query -> detected as dz, replied in MSA Arabic."""
    r = bot.answer("wesh 3andkom f les offres, nhab internet bezaf", idx)
    ok = r["lang"] == "dz" and _is_mostly_arabic(r["text"])
    return ok, f"lang={r['lang']} arabic_reply={_is_mostly_arabic(r['text'])}"


def t09_competitor(idx):
    """'offre Ooredoo?' -> polite refusal, Djezzy only."""
    r = bot.answer("C'est quoi les offres de Ooredoo ?", idx)
    t = r["text"].lower()
    ok = r["route"] == "competitor" and "djezzy" in t
    return ok, f"route={r['route']}"


def t10_out_of_domain(idx):
    """'la météo?' -> real out-of-domain decline (the deterministic refusal path)."""
    r = bot.answer("Quelle est la météo à Alger demain ?", idx)
    # A genuine decline: the OOD guard returned empty, so the bot served the
    # canned no-context refusal — NOT an LLM-generated answer we hope declined.
    is_real_decline = r["route"] == "no_context"
    no_fake_weather = "°" not in r["text"] and "degré" not in r["text"].lower()
    ok = is_real_decline and no_fake_weather
    return ok, f"route={r['route']}"


def t11_context_followup(idx):
    """Ask about Legend, then 'c'est combien?' still refers to Legend."""
    history = []
    r1 = bot.answer("Parle-moi de l'offre Legend", idx, history)
    history += [{"role": "user", "content": "Parle-moi de l'offre Legend"},
                {"role": "assistant", "content": r1["text"]}]
    r2 = bot.answer("c'est combien ?", idx, history)
    ok = "legend" in r2["text"].lower() and len(_prices(r2["text"])) >= 1
    return ok, f"followup_mentions_legend={'legend' in r2['text'].lower()}"


def t12_history_window(idx):
    """6 questions in a row; the 7th still answers correctly."""
    history = []
    warmups = ["Bonjour", "Vous avez des forfaits internet ?", "Et pour les jeunes ?",
               "C'est quoi iZZY ?", "Et Zid ?", "Merci"]
    for q in warmups:
        r = bot.answer(q, idx, history)
        history += [{"role": "user", "content": q},
                    {"role": "assistant", "content": r["text"]}]
    r7 = bot.answer("Quel est le prix de l'offre Legend ?", idx, history)
    ok = "legend" in r7["text"].lower() and len(_prices(r7["text"])) >= 1
    return ok, f"turn7_ok={ok} history_len={len(history)}"


def t13_ocr_offer(idx):
    """An image-only offer returns its price (proves OCR ingestion)."""
    # find a chunk that came from OCR and carries a price, then ask about it
    ocr_priced = None
    try:
        for d in idx.docstore._dict.values():
            if d.metadata.get("has_ocr") and _chunk_price(d.page_content):
                ocr_priced = d
                break
    except Exception:
        pass
    if ocr_priced is None:
        return False, "no OCR-sourced priced chunk in index (OCR found nothing?)"
    # ask using the offer name nearest the OCR text if present, else generic
    from data.lexicon import OFFER_NAMES
    name = next((n for n in OFFER_NAMES if n in ocr_priced.page_content.lower()),
                "cette offre")
    r = bot.answer(f"Quel est le prix de {name} ?", idx)
    ok = len(_prices(r["text"])) >= 1
    return ok, f"asked='{name}' got_price={_prices(r['text'])}"


def t14_deep_url_discovered(idx):
    """After scraping, a deep offer URL was discovered automatically."""
    urls = set()
    try:
        for d in idx.docstore._dict.values():
            urls.add(d.metadata.get("source_url", ""))
    except Exception:
        pass
    # a "deep" URL = path depth >= 3 segments under an offers/services section
    deep = [u for u in urls
            if u.count("/") >= 5 and ("offre" in u.lower() or "service" in u.lower())]
    ok = len(deep) >= 1
    return ok, f"deep_urls={len(deep)} example={deep[0] if deep else None}"


SCENARIOS = [
    ("01", "catalogue lists >=4 offers+prices", "fr", "catalogue",    t01_catalogue),
    ("02", "budget 500 shows only <=500",       "fr", "budget",       t02_budget),
    ("03", "named Campuce, no mixing",          "fr", "named-offer",  t03_named_campuce),
    ("04", "comparison Legend vs iZZY",         "fr", "named-offer",  t04_comparison),
    ("05", "roaming France not national",       "fr", "roaming",      t05_roaming),
    ("06", "English query -> English reply",    "en", "language",     t06_english),
    ("07", "Arabic query -> Arabic reply",      "ar", "language",     t07_arabic),
    ("08", "Darija -> MSA Arabic reply",        "dz", "language",     t08_darija),
    ("09", "Ooredoo -> competitor refusal",     "fr", "competitor",   t09_competitor),
    ("10", "weather -> out-of-domain refusal",  "fr", "out-of-domain", t10_out_of_domain),
    ("11", "context follow-up keeps Legend",    "fr", "context",      t11_context_followup),
    ("12", "history window, 7th still answers", "fr", "context",      t12_history_window),
    ("13", "image-only offer returns price",    "fr", "named-offer",  t13_ocr_offer),
    ("14", "deep URL auto-discovered",          "fr", "coverage",     t14_deep_url_discovered),
]


# ===========================================================================
# runner
# ===========================================================================
def run_all(idx) -> list:
    """Run every scenario against index `idx`; return a list of result dicts."""
    results = []
    for sid, name, lang, route, fn in SCENARIOS:
        try:
            passed, note = fn(idx)
        except Exception as e:
            passed, note = False, f"EXCEPTION: {e}"
        results.append({"id": sid, "name": name, "lang": lang,
                        "route": route, "passed": passed, "note": note})
        flag = "PASS" if passed else "FAIL"
        print(f"[{flag}] {sid} {name}  —  {note}")
    return results


def _summary(results):
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    print("\n" + "=" * 60)
    print(f"OVERALL: {passed}/{total} passed ({100*passed//total}%)")
    # by language
    print("\nBy language:")
    for lang in ("fr", "en", "ar", "dz"):
        sub = [r for r in results if r["lang"] == lang]
        if sub:
            p = sum(1 for r in sub if r["passed"])
            print(f"  {lang}: {p}/{len(sub)}")
    # by route
    print("\nBy route:")
    routes = sorted({r["route"] for r in results})
    for route in routes:
        sub = [r for r in results if r["route"] == route]
        p = sum(1 for r in sub if r["passed"])
        print(f"  {route}: {p}/{len(sub)}")
    print("=" * 60)
    return passed, total


def main():
    logging.basicConfig(level=logging.WARNING)
    import app
    app.boot()
    idx = app.STATE["index"]
    if idx is None:
        print("No index available — run the scraper/indexer first.")
        sys.exit(1)
    results = run_all(idx)
    passed, total = _summary(results)
    print(f"\nLatency records collected this run: {len(bot.LATENCY)}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
