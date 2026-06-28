"""
score_catalog.py — score the auto-extracted catalogs against the hand-verified GOLD.

After `build_catalog.py` writes offers.generated.json / roaming.generated.json, this
diffs them against the gold offers.json / roaming.json and prints, per offer:
  * MATCH        — same set of tier prices
  * MISSING      — a real (gold) tier the extractor dropped
  * PHANTOM      — a price the extractor invented that the gold doesn't have
plus an overall score. No API key needed — pure file comparison.

A high score on a BIG open Qwen proves the EXTRACTION METHOD is sound (the bug was the
regex, not the data). Then we check whether the local 7B can match it, or whether Djezzy
must run a bigger Qwen server-side.

Run:  python score_catalog.py
"""

import json
import sys

import config


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _prices(tiers):
    from collections import Counter
    return Counter(t["price"] for t in tiers or [])


def _diff(gold_tiers, gen_tiers):
    g, p = _prices(gold_tiers), _prices(gen_tiers)
    missing = list((g - p).elements())     # in gold, not generated
    phantom = list((p - g).elements())     # in generated, not gold
    return missing, phantom


def _score_section(title, gold_items, gen_by_key, key, tier_getters):
    print(f"\n=== {title} ===")
    total_gold = total_match = total_phantom = 0
    perfect = 0
    for item in gold_items:
        k = item[key]
        gen = gen_by_key.get(k)
        gold_tiers = [t for getter in tier_getters for t in (item.get(getter) or [])]
        if gen is None:
            print(f"  [NO OUTPUT] {item.get('name', k)}")
            total_gold += len(gold_tiers)
            continue
        gen_tiers = [t for getter in tier_getters for t in (gen.get(getter) or [])]
        missing, phantom = _diff(gold_tiers, gen_tiers)
        total_gold += len(gold_tiers)
        total_match += len(gold_tiers) - len(missing)
        total_phantom += len(phantom)
        if not missing and not phantom:
            perfect += 1
            print(f"  MATCH        {item.get('name', k)}  ({len(gold_tiers)} tiers)")
        else:
            bits = []
            if missing:
                bits.append(f"MISSING {missing}")
            if phantom:
                bits.append(f"PHANTOM {phantom}")
            print(f"  DIFF         {item.get('name', k)}: " + " | ".join(bits))
    recall = 100 * total_match / total_gold if total_gold else 100
    print(f"  -> {perfect}/{len(gold_items)} offers perfect | "
          f"tier recall {recall:.0f}% | {total_phantom} phantom price(s)")
    return perfect == len(gold_items) and total_phantom == 0


def main():
    ok = True
    try:
        gold_o = _load(config.OFFERS_JSON)["offers"]
        gen_o = {o["slug"]: o for o in _load(config.OFFERS_JSON.replace(".json", ".generated.json"))["offers"]}
        ok &= _score_section("OFFERS", gold_o, gen_o, "slug", ["tiers"])
    except FileNotFoundError:
        print("offers.generated.json not found — run build_catalog.py first.")
        ok = False
    try:
        gold_r = _load(config.ROAMING_JSON)["destinations"]
        gen_r = {d["slug"]: d for d in _load(config.ROAMING_JSON.replace(".json", ".generated.json"))["destinations"]}
        ok &= _score_section("ROAMING", gold_r, gen_r, "slug", ["mixte", "internet"])
    except FileNotFoundError:
        print("roaming.generated.json not found — run build_catalog.py first.")
        ok = False
    print("\n" + ("PASS — extractor matches gold." if ok else "REVIEW — see diffs above."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
