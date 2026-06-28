"""
data/catalog.py — the curated offers catalog (offers.json) as the source of truth
for the PRICED routes (named / catalogue / budget / comparison).

WHY THIS EXISTS
    The scraped offer pages mix real subscription tiers with recharge options,
    crédit/bonus amounts and per-unit tariff tables on the same page. The regex
    tier-parsing in retriever.py cannot separate them reliably, which is what made
    the bot answer "Legend = 100 DA", miss offers, and hallucinate prices. This
    catalog is a hand-verified, structured record per gamme; the priced routes build
    their LLM context straight from it, so there is nothing left to mis-parse.
    NORMAL / ROAMING / general questions keep using the scraped FAISS pages.

DESIGN
    * Pure stdlib — no langchain / numpy / torch — so the test suite and any offline
      tool can load and query the catalog with zero heavy deps. retriever.py wraps the
      plain text these functions return into LangChain Documents.
    * Only STRUCTURED fields reach the model (summary, eligibility, activation, tiers,
      fees, extra_offers). The free-form `notes` field is CURATOR-ONLY and is never
      shown — it holds verification commentary ("vérifié sur capture…") that must not
      leak into a customer answer.
    * Records flagged "verify": true are still served (they are the best data we have),
      but the catalog is meant to be regenerated/checked against the public site.
"""

import json
import unicodedata

import config

# Cache: parsed once per process. Tests can reset via _reset() if they patch the file.
_OFFERS = None
_INDEX = None
_ROAMING = None
_ROAMING_INDEX = None

# type -> human label shown in the offer header.
_TYPE_LABEL = {
    "prepaid": "prépayée",
    "postpaid": "postpayée / abonnement",
    "service": "service",
    "info": "information",
}


def _norm(s: str) -> str:
    """Fold a name/slug to a comparison key: lowercase, no accents, no separators.

    So the canonical names the retriever detects ("legend max", "flexy net",
    "djezzy 5g", "campuce") match the catalog's name/slug ("legend-max",
    "flexy-net", "djezzy-5g", "campuce") regardless of spaces or hyphens.
    """
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    for ch in (" ", "-", "_"):
        s = s.replace(ch, "")
    return s


def _reset():
    """Drop the in-memory cache (used by tests after patching the JSON files)."""
    global _OFFERS, _INDEX, _ROAMING, _ROAMING_INDEX
    _OFFERS, _INDEX, _ROAMING, _ROAMING_INDEX = None, None, None, None


def records() -> list:
    """All offer records from offers.json (cached). Empty list if unreadable."""
    global _OFFERS
    if _OFFERS is None:
        try:
            with open(config.OFFERS_JSON, encoding="utf-8") as fh:
                _OFFERS = json.load(fh).get("offers", [])
        except (OSError, ValueError):
            _OFFERS = []
    return _OFFERS


def _index() -> dict:
    """Map every name/slug fold-key -> record (cached)."""
    global _INDEX
    if _INDEX is None:
        idx = {}
        for rec in records():
            for key in (rec.get("name", ""), rec.get("slug", "")):
                k = _norm(key)
                if k:
                    idx.setdefault(k, rec)
        _INDEX = idx
    return _INDEX


def by_name(name: str):
    """The catalog record for a detected offer name (canonical or slug), or None."""
    return _index().get(_norm(name))


def _tiers_sorted(rec: dict) -> list:
    """The record's real subscription tiers, cheapest-first (numeric price only)."""
    tiers = [t for t in (rec.get("tiers") or [])
             if isinstance(t.get("price"), (int, float))]
    return sorted(tiers, key=lambda t: t["price"])


def starting_price(rec: dict):
    """The cheapest tier price, or None for a service / info record with no tiers."""
    tiers = _tiers_sorted(rec)
    return tiers[0]["price"] if tiers else None


def _header(rec: dict) -> str:
    name = rec.get("name", "").strip()
    head = name if _norm(name).startswith("djezzy") else f"Djezzy {name}"
    label = _TYPE_LABEL.get(rec.get("type", ""), "")
    return f"Offre : {head}" + (f" ({label})" if label else "")


def _common_body(rec: dict, lines: list, tiers: list):
    """Append the non-tier structured fields (fees, extras, activation, footers)."""
    for fee_block, intro in ((rec.get("fees"), "Frais de transfert (ce ne sont PAS des forfaits) :"),):
        if fee_block:
            lines.append(intro)
            for f in fee_block:
                lines.append(f"- {f.get('price')} DA : {f.get('details', '')}")
    for extra in rec.get("extra_offers") or []:
        lines.append(f"{extra.get('name', 'Option')} : {extra.get('details', '')}")
    if rec.get("activation"):
        lines.append("Activation : " + rec["activation"])
    if rec.get("type") == "info" and not tiers:
        lines.append("Aucun tarif public n'est communiqué pour cette offre ; "
                     "les forfaits se choisissent dans l'application Djezzy.")
    if rec.get("type") == "service" and not tiers and not rec.get("fees"):
        lines.append("Service — pas de tarif d'abonnement.")
    if rec.get("url"):
        lines.append("Page : " + rec["url"])


def format_offer(rec: dict) -> str:
    """Full clean context block for the NAMED / COMPARISON routes (all tiers)."""
    lines = [_header(rec)]
    if rec.get("summary"):
        lines.append(rec["summary"])
    if rec.get("eligibility"):
        lines.append("Éligibilité : " + rec["eligibility"])
    tiers = _tiers_sorted(rec)
    if tiers:
        lines.append("Tarifs (du moins cher au plus cher) :")
        for t in tiers:
            lines.append(f"- {t['price']} DA : {t.get('details', '')}")
    _common_body(rec, lines, tiers)
    return "\n".join(lines)


def format_budget(rec: dict, budget: int):
    """(text, cheapest_kept_price) keeping only tiers <= budget, or (None, None).

    Built straight from the structured tiers, so no crédit/bonus amount or per-unit
    rate can ever be mistaken for an affordable price (the budget-leak bugs).
    """
    affordable = [t for t in _tiers_sorted(rec) if t["price"] <= budget]
    if not affordable:
        return None, None
    lines = [_header(rec), "Forfaits dans votre budget (du moins cher au plus cher) :"]
    for t in affordable:
        lines.append(f"- {t['price']} DA : {t.get('details', '')}")
    if rec.get("url"):
        lines.append("Page : " + rec["url"])
    return "\n".join(lines), affordable[0]["price"]


def catalogue_line(rec: dict) -> str:
    """One menu line for the CATALOGUE route: name + starting price (or kind)."""
    name = rec.get("name", "")
    sp = starting_price(rec)
    if sp is not None:
        return f"{name} : à partir de {sp} DA"
    if rec.get("type") == "service":
        return f"{name} : service"
    if rec.get("type") == "info":
        return f"{name} : information (pas de tarif public)"
    return name


# ===========================================================================
# Roaming catalog (roaming.json) — source of truth for the ROAMING route
# ===========================================================================
# Djezzy roaming pages mix two product families ("Internet & Voix" and "Internet
# seul") and use inconsistent price layouts ("Pour 2000 DA" on some pages, plain
# "2000 DA" on the Hadj/Omra page), which the regex tier-parser cannot align — the
# cause of the garbled / hallucinated roaming answers. Each destination is curated
# here once, so the roaming route answers from clean structured tiers.

def roaming_destinations() -> list:
    """All roaming destination records from roaming.json (cached)."""
    global _ROAMING
    if _ROAMING is None:
        try:
            with open(config.ROAMING_JSON, encoding="utf-8") as fh:
                _ROAMING = json.load(fh).get("destinations", [])
        except (OSError, ValueError):
            _ROAMING = []
    return _ROAMING


def _roaming_index() -> dict:
    """Map every marker / slug / name fold-key -> destination record (cached)."""
    global _ROAMING_INDEX
    if _ROAMING_INDEX is None:
        idx = {}
        for dest in roaming_destinations():
            keys = list(dest.get("markers", [])) + [dest.get("slug", ""), dest.get("name", "")]
            for key in keys:
                k = _norm(key)
                if k:
                    idx.setdefault(k, dest)
        _ROAMING_INDEX = idx
    return _ROAMING_INDEX


def roaming_for_markers(markers: list):
    """The roaming destination record matching any of `markers`, or None.

    `markers` are the tokens the retriever's _roaming_markers() returns (e.g.
    ["egypte"] or ["hadj", "omra"]); the generic ["roaming"] matches nothing here
    so the route falls back to the scraped general roaming page.
    """
    idx = _roaming_index()
    for m in markers or []:
        dest = idx.get(_norm(m))
        if dest is not None:
            return dest
    return None


def format_roaming(dest: dict) -> str:
    """Full clean context block for one roaming destination (both families)."""
    lines = [f"Roaming {dest.get('name', '')} ({dest.get('zone', '')})"]
    mixte = sorted(dest.get("mixte", []), key=lambda t: t["price"])
    internet = sorted(dest.get("internet", []), key=lambda t: t["price"])
    if mixte:
        lines.append("Forfaits Internet & Voix (du moins cher au plus cher) :")
        for t in mixte:
            lines.append(f"- {t['price']} DA : {t.get('details', '')}")
    if internet:
        lines.append("Forfaits Internet seul (du moins cher au plus cher) :")
        for t in internet:
            lines.append(f"- {t['price']} DA : {t.get('details', '')}")
    if dest.get("welcome"):
        lines.append("SIM de bienvenue : " + dest["welcome"])
    if dest.get("info"):                       # customer-facing; `notes` is curator-only
        lines.append(dest["info"])
    if dest.get("url"):
        lines.append("Page : " + dest["url"])
    return "\n".join(lines)
