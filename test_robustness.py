"""
test_robustness.py — extended STRESS suite (separate from the acceptance suite).

`test_scenarios.py` is the formal *acceptance* contract (14 scenarios, the number
the thesis reports). THIS file is the *robustness* suite: ~16 harder, mostly
cross-lingual scenarios that probe the routes and the headline contributions far
more aggressively. Failures here are engineering findings to fix, NOT a broken
acceptance contract — so this suite is allowed to surface weak spots.

What it adds over the acceptance suite:
  * every major route exercised in Arabic / Darija / English, not French only;
  * the contributions that were previously untested end to end:
      - the competitor firewall's ALLOW case ("iZZY calls Mobilis" must NOT refuse),
      - Hajj / Umrah roaming,
      - cheapest-first tier ORDERING (ascending, not merely present),
      - anti-hallucination on a non-existent offer,
      - the out-of-domain FALSE-REFUSAL guard (a short real question must still answer);
  * budget hardening (other ceilings, the per-unit-rate trap, an impossible budget).

Usage
-----
    python test_robustness.py                 # boot index, run all, print summary
    from test_robustness import run_all       # returns the results list
    # In the Colab notebook (after `store` is built):
    #     import test_robustness; test_robustness._summary(test_robustness.run_all(store))

Each result: {id, name, lang, route, passed, note}. Same shape as the acceptance
suite, so the same reporting code works.
"""

import re
import sys
import logging

import config
import bot
import retriever
from retriever import smart_retrieve, _chunk_price

# reuse the acceptance suite's vetted helpers so the two suites judge identically
from test_scenarios import (
    _prices, _offer_prices, _count_offer_mentions,
    _is_mostly_arabic, _is_english,
)

logger = logging.getLogger("djezzybot.robustness")

_OTHER_GAMMES = ("legend", "izzy", "cam puce", "campuce", "zid", "confort", "flexy", "hayla", "3ayla")
_DECLINE_WORDS = ("pas", "aucun", "n'existe", "n existe", "introuvable", "désolé",
                  "desole", "ne dispose", "non disponible", "trouvé", "trouve",
                  "sorry", "don't", "do not", "not available", "ليس", "لا يوجد", "غير")


def _ascending(nums) -> bool:
    """True if the numeric list is non-decreasing (cheapest-first ordering)."""
    return all(a <= b for a, b in zip(nums, nums[1:]))


def _others_present(text: str, keep: str):
    """Gamme names other than `keep` that leaked into the reply."""
    t = text.lower()
    keep = keep.lower()
    return [n for n in _OTHER_GAMMES if n in t and n not in keep and keep not in n]


# ===========================================================================
# Group A — every route, but in Arabic / Darija / English (not French)
# ===========================================================================
def r01_budget_arabic(idx):
    """Budget intent in Arabic: only offers <= budget in the answer."""
    budget = 1000
    r = bot.answer("عندي 1000 دينار، شنو تنصحوني؟", idx)
    op = _offer_prices(r["text"])
    ok = _is_mostly_arabic(r["text"]) and len(op) > 0 and all(p <= budget for p in op)
    return ok, f"lang={r['lang']} route={r['route']} offer_prices={op}"


def r02_named_arabic(idx):
    """Named offer asked in Arabic ('Legend') -> Legend, with a price, in Arabic."""
    r = bot.answer("قولي على عرض ليجند بالتفصيل", idx)
    t = r["text"].lower()
    ok = "legend" in t and _is_mostly_arabic(r["text"]) and len(_prices(r["text"])) >= 1
    return ok, f"lang={r['lang']} legend={'legend' in t} prices={_prices(r['text'])}"


def r03_catalogue_arabic(idx):
    """Catalogue asked in Arabic -> several offers/prices, Arabic reply."""
    r = bot.answer("شنو هي العروض المتوفرة عندكم؟", idx)
    ok = _is_mostly_arabic(r["text"]) and (_count_offer_mentions(r["text"]) >= 3
                                           or len(_prices(r["text"])) >= 3)
    return ok, f"lang={r['lang']} offers={_count_offer_mentions(r['text'])} prices={len(_prices(r['text']))}"


def r04_competitor_arabic(idx):
    """Competitor named in Arabic (Ooredoo = أوريدو) -> refusal, Djezzy only."""
    r = bot.answer("واش هي عروض أوريدو؟", idx)
    ok = r["route"] == "competitor"
    return ok, f"route={r['route']}"


def r05_roaming_hajj_arabic(idx):
    """Hajj roaming in Arabic (حج) -> roaming context, not a national tariff."""
    docs = smart_retrieve("بغيت رومينغ باش نروح للحج", "ar", idx)
    if docs == config.COMPETITOR_SENTINEL:
        return False, "unexpected competitor route"
    blob = " ".join(d.page_content.lower() + " " +
                    d.metadata.get("source_url", "").lower() for d in docs)
    ok = "roaming" in blob or "حج" in blob or "hajj" in blob or "omra" in blob or "عمرة" in blob
    return ok, f"roaming_ctx={'roaming' in blob} n_docs={len(docs)}"


def r06_ood_english(idx):
    """Out-of-domain in English (weather) -> declined, no real weather."""
    r = bot.answer("What's the weather in Algiers tomorrow?", idx)
    is_decline = r["route"] in ("no_context", "normal")
    low = r["text"].lower()
    weather = ("°", "degree", "celsius", "sunny", "rain", "cloud", "wind", "humidity",
               "forecast", "temperature")
    ok = is_decline and not any(w in low for w in weather)
    return ok, f"route={r['route']}"


# ===========================================================================
# Group B — previously-untested contributions
# ===========================================================================
def r07_mobilis_allow_case(idx):
    """The firewall's ALLOW case: 'iZZY calls to Mobilis' must NOT be refused.

    Mobilis is a call DESTINATION here, not the subject; the question is about a
    Djezzy offer (iZZY), so it must reach a normal answer, not the competitor
    refusal. This is the precise nuance of contribution #10.
    """
    r = bot.answer("Avec iZZY, combien coûtent les appels vers Mobilis ?", idx)
    ok = r["route"] != "competitor"
    return ok, f"route={r['route']} (must NOT be competitor)"


def r08_named_tiers_ascending(idx):
    """Cam Puce's tiers must come back cheapest-first (ascending), not just present.

    Tests the deterministic ordering contribution at the ANSWER level: the offer
    prices that appear should be in non-decreasing order.
    """
    r = bot.answer("Donne-moi tous les paliers de l'offre Cam Puce avec leurs prix", idx)
    pr = _prices(r["text"])
    ok = len(pr) >= 2 and _ascending(pr)
    return ok, f"prices={pr} ascending={_ascending(pr) if pr else None}"


def r09_nonexistent_offer(idx):
    """A non-existent offer ('Diamond Premium') must not get an invented price.

    Grounding (rule 2): the bot should decline / say it has no such offer rather
    than fabricate a price for a name that exists nowhere in the data.
    """
    r = bot.answer("Parle-moi de l'offre Diamond Premium de Djezzy", idx)
    t = r["text"].lower()
    declined = any(w in t for w in _DECLINE_WORDS)
    # if it did NOT decline, it must at least not have invented a price next to "diamond"
    invented = ("diamond" in t) and len(_prices(r["text"])) >= 1
    ok = declined or not invented
    return ok, f"route={r['route']} declined={declined} invented_price={invented}"


def r10_short_real_not_refused(idx):
    """OOD false-refusal guard: a SHORT real question must still be answered.

    The calibration finding is that short real questions (esp. Darija) score low;
    the design must NOT refuse them. 'ch7al izzy?' is a real price question.
    """
    r = bot.answer("ch7al izzy ?", idx)
    refused = r["route"] in ("no_context", "competitor")
    answered = (not refused) and (len(r["text"].strip()) > 0)
    return answered, f"route={r['route']} refused={refused}"


def r11_competitor_mobilis(idx):
    """Competitor subject = Mobilis (not just Ooredoo) -> refusal."""
    r = bot.answer("C'est quoi les offres de Mobilis ?", idx)
    ok = r["route"] == "competitor"
    return ok, f"route={r['route']}"


# ===========================================================================
# Group C — budget hardening (the weakest route)
# ===========================================================================
def r12_budget_other_ceiling(idx):
    """A different budget ceiling (200 DA), French -> only offers <= 200."""
    budget = 200
    r = bot.answer("j'ai seulement 200 DA, quelles offres ?", idx)
    op = _offer_prices(r["text"])
    ok = len(op) > 0 and all(p <= budget for p in op)
    return ok, f"route={r['route']} offer_prices={op}"


def r13_budget_no_rate_leak(idx):
    """Per-unit-rate trap: a budget answer must not present a '/unit' tariff as a price.

    The reply must contain no 'N DA / SMS|Mo|min...' pattern dressed up as an offer.
    """
    r = bot.answer("j'ai 500 DA, propose-moi des forfaits", idx)
    leak = re.search(r"\d+\s*da\s*/\s*(sms|mo|mb|go|min|sec|message|appel)",
                     r["text"].lower())
    ok = leak is None
    return ok, f"route={r['route']} rate_leak={'yes' if leak else 'no'}"


def r14_budget_impossible(idx):
    """An impossible budget (10 DA) -> no offer priced above it, no fabrication."""
    budget = 10
    r = bot.answer("j'ai 10 DA, qu'est-ce que je peux avoir ?", idx)
    op = _offer_prices(r["text"])
    ok = all(p <= budget for p in op)        # ideally empty or a graceful 'nothing fits'
    return ok, f"route={r['route']} offer_prices={op}"


# ===========================================================================
# Group D — named-offer breadth
# ===========================================================================
def r15_named_zid(idx):
    """Another named offer ('Zid') -> Zid, no other gamme mixed in."""
    r = bot.answer("Parle-moi de l'offre Zid", idx)
    t = r["text"].lower()
    has_zid = "zid" in t
    others = _others_present(t, "zid")
    ok = has_zid and not others
    return ok, f"zid={has_zid} other_gammes={others}"


def r16_multitier_ascending(idx):
    """A multi-tier offer (Flexy) returns its tiers cheapest-first (ascending)."""
    r = bot.answer("Quels sont les montants Flexy disponibles ?", idx)
    pr = _prices(r["text"])
    ok = len(pr) >= 2 and _ascending(pr)
    return ok, f"prices={pr} ascending={_ascending(pr) if pr else None}"


SCENARIOS = [
    # Group A — cross-lingual route coverage
    ("r01", "budget intent in Arabic",            "ar", "budget",       r01_budget_arabic),
    ("r02", "named offer in Arabic (Legend)",     "ar", "named-offer",  r02_named_arabic),
    ("r03", "catalogue in Arabic",                "ar", "catalogue",    r03_catalogue_arabic),
    ("r04", "competitor in Arabic (Ooredoo)",     "ar", "competitor",   r04_competitor_arabic),
    ("r05", "Hajj roaming in Arabic",             "ar", "roaming",      r05_roaming_hajj_arabic),
    ("r06", "out-of-domain in English",           "en", "out-of-domain", r06_ood_english),
    # Group B — untested contributions
    ("r07", "firewall ALLOW: iZZY->Mobilis",      "fr", "competitor",   r07_mobilis_allow_case),
    ("r08", "Cam Puce tiers ascending",           "fr", "named-offer",  r08_named_tiers_ascending),
    ("r09", "non-existent offer, no fabrication", "fr", "named-offer",  r09_nonexistent_offer),
    ("r10", "short real question not refused",    "dz", "out-of-domain", r10_short_real_not_refused),
    ("r11", "competitor subject = Mobilis",       "fr", "competitor",   r11_competitor_mobilis),
    # Group C — budget hardening
    ("r12", "budget other ceiling (200)",         "fr", "budget",       r12_budget_other_ceiling),
    ("r13", "budget: no per-unit-rate leak",      "fr", "budget",       r13_budget_no_rate_leak),
    ("r14", "budget impossible (10 DA)",          "fr", "budget",       r14_budget_impossible),
    # Group D — named-offer breadth
    ("r15", "named offer Zid, no mixing",         "fr", "named-offer",  r15_named_zid),
    ("r16", "Flexy multi-tier ascending",         "fr", "named-offer",  r16_multitier_ascending),
]


# ===========================================================================
# runner (same shape as test_scenarios)
# ===========================================================================
def run_all(idx) -> list:
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
    print(f"ROBUSTNESS: {passed}/{total} passed ({100*passed//total}%)")
    print("\nBy language:")
    for lang in ("fr", "en", "ar", "dz"):
        sub = [r for r in results if r["lang"] == lang]
        if sub:
            p = sum(1 for r in sub if r["passed"])
            print(f"  {lang}: {p}/{len(sub)}")
    print("\nBy route:")
    for route in sorted({r["route"] for r in results}):
        sub = [r for r in results if r["route"] == route]
        p = sum(1 for r in sub if r["passed"])
        print(f"  {route}: {p}/{len(sub)}")
    print("\nFailures to investigate:")
    fails = [r for r in results if not r["passed"]]
    for r in fails:
        print(f"  [{r['id']}] {r['name']}  —  {r['note']}")
    if not fails:
        print("  (none)")
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
    # robustness suite is exploratory: exit 0 regardless, the findings are the point
    sys.exit(0)


if __name__ == "__main__":
    main()
