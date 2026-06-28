"""
test_catalog.py — pure-Python proof that the priced routes now answer from the
curated catalog (data/offers.json), not from brittle scraped-chunk parsing.

No GPU / FAISS / langchain-community needed: the catalog routes in smart_retrieve
return BEFORE any vector-store call, so we pass vector_db=None. Run from the project
root:  python test_catalog.py
"""

import re
import sys

import retriever
from data import catalog

FAILS = []


def check(cond, msg):
    print(("  ok  " if cond else "FAIL  ") + msg)
    if not cond:
        FAILS.append(msg)


def doc_text(docs):
    return "\n".join(d.page_content for d in docs)


def prices_in(text):
    """Every '<n> DA' amount in the text, as ints (ignoring 'Go', 'min', etc.)."""
    return [int(m.replace(" ", "")) for m in re.findall(r"(\d[\d ]*)\s*DA", text)]


print("=== catalog loads & maps every detected name ===")
recs = catalog.records()
check(len(recs) == 12, f"12 offers loaded (got {len(recs)})")
for name in ["legend", "legend max", "legend pro", "campuce", "izzy", "zid",
             "confort", "3ayla", "facebook flex", "flexy net", "flexy", "djezzy 5g"]:
    check(catalog.by_name(name) is not None, f"by_name maps '{name}'")

print("\n=== NAMED: Legend leads with the real range, not '100 DA only' ===")
docs = retriever.smart_retrieve("parle-moi de l'offre Legend", "fr", None)
txt = doc_text(docs)
check(all(d.metadata.get("catalog") for d in docs), "named served from catalog")
for p in (100, 1000, 1500, 2000, 2500, 3000, 4000):
    check(p in prices_in(txt), f"Legend context contains the {p} DA tier")
check("200 Go" in txt, "Legend context mentions 200 Go (top tier)")
# the bug was a phantom price: nothing outside the real tier set should appear
check(set(prices_in(txt)) <= {100, 300, 1000, 1500, 2000, 2500, 3000, 4000},
      "Legend context has no phantom price (300 = real crédit on the 100 DA tier)")

print("\n=== NAMED: 5G is info-only, zero invented price ===")
docs = retriever.smart_retrieve("parle-moi de Djezzy 5G", "fr", None)
txt = doc_text(docs)
check(all(d.metadata.get("catalog") for d in docs), "5G served from catalog")
check(prices_in(txt) == [], "5G context carries NO price")
check("application" in txt.lower(), "5G context points to the app for plans")

print("\n=== BUDGET: 'j'ai 500 DA' returns only tiers <= 500, cheapest first ===")
docs = retriever.smart_retrieve("j'ai 500 DA, quelle offre ?", "fr", None)
txt = doc_text(docs)
check(len(docs) > 0, "budget returned at least one affordable offer")
check(all(p <= 500 for p in prices_in(txt)), "no tier above 500 DA leaked in")

print("\n=== BUDGET: tight 60 DA only surfaces the 50 DA tiers ===")
docs = retriever.smart_retrieve("j'ai seulement 60 DA", "fr", None)
txt = doc_text(docs)
check(prices_in(txt) and max(prices_in(txt)) <= 60, "60 DA budget keeps only <=60 DA tiers")

print("\n=== CATALOGUE: every gamme listed once, priced gammes first ===")
docs = retriever.smart_retrieve("quelles sont vos offres ?", "fr", None)
txt = doc_text(docs)
check(len(docs) == 12, f"catalogue lists all 12 gammes (got {len(docs)})")
check("Legend : à partir de 100 DA" in txt, "Legend menu line shows starting price 100 DA")
check("Djezzy 5G : information (pas de tarif public)" in txt, "5G listed without a price")
def line_price(text):
    m = re.search(r"à partir de (\d+) DA", text)
    return int(m.group(1)) if m else None
seq = [line_price(d.page_content) for d in docs]
priced_seq = [p for p in seq if p is not None]
check(priced_seq == sorted(priced_seq), "priced gammes are ordered cheapest-first")
first_none = next((i for i, p in enumerate(seq) if p is None), len(seq))
check(all(p is None for p in seq[first_none:]), "services/info listed after priced gammes")

print("\n=== COMPARISON: Legend vs iZZY → both, cheapest offer first ===")
docs = retriever.smart_retrieve("différence entre Legend et iZZY", "fr", None)
txt = doc_text(docs)
check(len(docs) == 2, "comparison returns exactly the two named offers")
check("Legend" in txt and "iZZY" in txt, "both Legend and iZZY present")

print("\n=== ROAMING: known destinations served from the catalog ===")
dests = catalog.roaming_destinations()
check(len(dests) == 6, f"6 roaming destinations loaded (got {len(dests)})")

docs = retriever.smart_retrieve("je voyage en France, quel forfait roaming ?", "fr", None)
txt = doc_text(docs)
check(all(d.metadata.get("catalog") for d in docs), "France roaming served from catalog")
check("Internet & Voix" in txt and "Internet seul" in txt, "France shows both product families")
for p in (1000, 2000, 5000):
    check(p in prices_in(txt), f"France roaming context has the {p} DA mixte tier")

docs = retriever.smart_retrieve("forfait roaming pour l'Égypte", "fr", None)
check(set(prices_in(doc_text(docs))) >= {1000, 2000, 4000, 3000}, "Égypte roaming tiers present")

print("\n=== ROAMING: Hadj/Omra (the page the parser used to garble) ===")
docs = retriever.smart_retrieve("offre roaming pour le Hadj", "fr", None)
txt = doc_text(docs)
check(all(d.metadata.get("catalog") for d in docs), "Hadj served from catalog")
for p in (1000, 2000, 4000):
    check(p in prices_in(txt), f"Hadj/Omra context has the {p} DA tier")
check("Arabie Saoudite" in txt, "Hadj context states Saudi-Arabia-only")
check("bienvenue" in txt.lower() or "sim gratuite" in txt.lower(), "Hadj mentions the free welcome SIM")

print("\n=== ROAMING: phantom destinations no longer route to roaming ===")
check(retriever._roaming_markers("je voyage au Maroc") is None,
      "Maroc no longer a roaming destination")
check("dubai" not in (retriever._roaming_markers("roaming à Dubai") or []),
      "Dubai no longer a roaming destination")
check(catalog.roaming_for_markers(["maroc"]) is None, "no Maroc record in roaming catalog")

print("\n=== ROUTING unchanged (classify_route still correct) ===")
check(retriever.classify_route("parle-moi de Legend") == "named", "named route")
check(retriever.classify_route("j'ai 500 DA") == "budget", "budget route")
check(retriever.classify_route("quelles sont vos offres") == "catalogue", "catalogue route")
check(retriever.classify_route("différence entre Legend et iZZY") == "comparison", "comparison route")
check(retriever.classify_route("forfait roaming Égypte") == "roaming", "roaming route")

print("\n" + ("ALL PASS" if not FAILS else f"{len(FAILS)} FAILURE(S):\n  - " + "\n  - ".join(FAILS)))
sys.exit(1 if FAILS else 0)
