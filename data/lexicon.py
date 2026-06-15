"""
data/lexicon.py — Language & domain knowledge for the retriever.

Pure data + a couple of tiny pure functions, no heavy dependencies. This is the
"vocabulary" the smart retriever uses to understand what a user is asking before
it ever touches the vector store:

  * OFFER_NAMES        — canonical Djezzy offer/service names (for routing)
  * SYNONYMS           — query expansion (FR + Darija → canonical retrieval terms)
  * COMPETITORS        — trigger the competitor firewall (Latin + Arabic script)
  * ROAMING_TRIGGERS   — destination words → roaming markers (incl. Arabic)
  * BUDGET_TRIGGERS    — intent cues that mark a "I have X DA" question
  * CATALOGUE_TRIGGERS — "show me your offers" intent
  * COMPARISON_TRIGGERS— "difference between A and B" intent
  * DARIJA_WORDS       — Algerian Darija markers for language detection

Everything is lowercase; callers normalize the query to lowercase before matching.
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Canonical offer / service names
# ---------------------------------------------------------------------------
# Used for: catalogue enumeration, named-offer detection, comparison detection.
# Order matters for detection: multi-word / more-specific names come first so
# "legend max" is matched before the bare "legend".
# The real prepaid/postpaid FORFAIT gammes + the Flexy credit service (verified
# against the live crawl). "Flexy" is included because it is an everyday term in
# Algeria (topping up / transferring credit) that subscribers ask about by name;
# it is ordered AFTER "flexy net" so the more specific name matches first.
# Still deliberately NOT including generic words like "carte" (carte SIM) or
# "control" (the parental-control SERVICE) — those caused false catalogue matches.
OFFER_NAMES = [
    "legend max",
    "legend pro",
    "legend",
    "cam puce",
    "campuce",
    "izzy",
    "zid",
    "confort",
    "3ayla",
    "facebook flex",
    "flexy net",
    "flexy",
    "djezzy 5g",
]

# Offers that are postpaid / subscription plans. The retriever can use this to
# avoid surfacing a monthly subscription when the user just wants a prepaid top-up.
POSTPAID_OFFERS = ["legend max", "legend pro", "confort"]

# Content types beyond forfaits — Djezzy also sells physical phones and offers
# many services. These terms keep device/service questions IN domain (the OOD
# guard) and let callers tell content types apart; the pages are retrieved
# normally via FAISS (they're indexed like any other page).
DEVICE_TERMS = [
    "telephone", "portable", "smartphone", "mobile", "appareil",
    "zte", "blade", "nubia", "tecno", "spark",   # brands seen under /nos-mobiles/
]
SERVICE_TERMS = [
    "service", "controle parental", "e-flexy", "eflexy", "scoop", "fennec",
    "liste rouge", "double appel", "appel masque", "conference", "ranati",
    "facture", "changement de carte sim",
]

# ---------------------------------------------------------------------------
# Synonym / query expansion
# ---------------------------------------------------------------------------
# Maps a user phrase (FR or Darija) to extra retrieval terms. The expansion is
# appended to the query before embedding so FAISS sees Djezzy vocabulary even
# when the user used slang or a generic word. Keep expansions short and on-topic.
SYNONYMS = {
    # generic data / internet
    "internet": "data go mo connexion",
    "gigas": "go data volume",
    "giga": "go data volume",
    "data": "go internet volume",
    "connexion": "internet data go",
    # volume intent
    "beaucoup": "max grand volume go",
    "bezaf": "max grand volume go beaucoup",       # Darija: a lot
    "gros volume": "max grand 100 go 200 go",
    # price / budget vocabulary
    "prix": "tarif da forfait",
    "tarif": "prix da forfait",
    "combien": "prix tarif da",
    "chhal": "combien prix tarif da",              # Darija: how much
    "pas cher": "moins cher economique petit prix",
    # audience
    "jeunes": "campuce cam puce etudiant student",
    "étudiant": "campuce cam puce student",
    "etudiant": "campuce cam puce student",
    "social": "reseaux sociaux facebook instagram tiktok",
    "réseaux sociaux": "social facebook instagram tiktok",
    # offer-shaped intents
    "forfait": "offre legend izzy campuce zid confort",
    "offre": "forfait legend izzy campuce zid confort",
    "abonnement": "forfait postpaid legend max control",
    # roaming
    "voyage": "roaming etranger international",
    "étranger": "roaming international voyage",
    "etranger": "roaming international voyage",
    "à l'étranger": "roaming international",
    # Darija verbs → French (helps the multilingual embedder anchor)
    "nhab": "je veux offre forfait",
    "bghit": "je veux offre forfait",
    "3andi": "j'ai budget",
    "3tini": "donne moi montre",
    "wesh": "quels offres",
    "kayen": "y a-t-il disponible",
}

# ---------------------------------------------------------------------------
# Competitor firewall
# ---------------------------------------------------------------------------
# Any of these in the query => smart_retrieve returns the COMPETITOR sentinel
# and the bot replies with a polite "Djezzy only" refusal. Includes Arabic script.
# Competitor brand names. These are tricky: a Djezzy offer legitimately says
# "appels illimités vers Ooredoo et Mobilis", so a bare brand mention is NOT
# always a competitor question — the retriever decides using CALL_DEST_CUES.
COMPETITOR_BRANDS = [
    "ooredoo",
    "mobilis",
    "nedjma",      # legacy brand → Ooredoo
    "موبيليس",     # Mobilis (Arabic)
    "أوريدو",      # Ooredoo (Arabic)
    "اوريدو",      # Ooredoo (Arabic, alif variant)
]

# Generic "competition" words — always a competitor question (refuse).
COMPETITOR_GENERIC = [
    "concurrent",
    "concurrence",
    "competitor",
    "djezzy vs",
]

# Cues that, before a brand, mean it's a CALL DESTINATION inside a Djezzy offer
# ("appels vers Mobilis", "calls to Ooredoo") — not a question about that operator.
CALL_DEST_CUES = [
    "vers", "appeler", "appelle", "appel", "appels", "appeles",
    "call", "calls", " to ", "reseau", "operateur", "operateurs",
    "نحو", "الى", "إلى", "مكالمات",
]

# Backward-compatible flat list of everything that touches the firewall.
COMPETITORS = COMPETITOR_BRANDS + COMPETITOR_GENERIC

# ---------------------------------------------------------------------------
# Roaming destination map
# ---------------------------------------------------------------------------
# destination word found in the query -> markers used to locate roaming chunks.
# Hadj/Omra (Saudi pilgrimage) and the Arabic forms are deliberately included.
ROAMING_TRIGGERS = {
    "hadj": ["hadj", "omra"],
    "omra": ["omra", "hadj"],
    "pèlerinage": ["hadj", "omra"],
    "pelerinage": ["hadj", "omra"],
    "saoudite": ["hadj", "omra"],
    "mecque": ["hadj", "omra"],
    "حج": ["hadj", "omra"],          # Hadj (Arabic)
    "عمرة": ["omra", "hadj"],        # Omra (Arabic)
    "france": ["france"],
    "tunisie": ["tunisie"],
    "espagne": ["espagne"],
    "turquie": ["turquie"],
    "égypte": ["egypte"],
    "egypte": ["egypte"],
    "maroc": ["maroc"],
    "dubai": ["emirats", "dubai"],
    "emirats": ["emirats", "dubai"],
    # Arabic country names (a customer asking "عرض تاع مصر ?" = an Egypt ROAMING offer).
    # These only count as roaming when paired with a roaming/travel/offer cue (see
    # retriever._roaming_markers), so "ما هي عاصمة فرنسا؟" stays out-of-domain.
    "مصر": ["egypte"],
    "تونس": ["tunisie"],
    "فرنسا": ["france"],
    "تركيا": ["turquie"],
    "المغرب": ["maroc"],
    "اسبانيا": ["espagne"],
    "إسبانيا": ["espagne"],
    "السعودية": ["hadj", "omra"],
    "الإمارات": ["emirats", "dubai"],
    "دبي": ["emirats", "dubai"],
}
# Generic roaming words that, combined with a travel cue, also mean roaming.
ROAMING_GENERIC = ["roaming", "étranger", "etranger", "international"]
ROAMING_TRAVEL_CUES = ["voyage", "voyager", "pars", "partir", "à l'étranger", "abroad"]

# ---------------------------------------------------------------------------
# Budget intent
# ---------------------------------------------------------------------------
# A budget question needs BOTH an amount (parsed by the retriever) AND an intent
# cue from this list. FR + Darija (romanized) + Arabic script + EN.
BUDGET_TRIGGERS = [
    # French
    "j'ai", "j ai", "budget", "moins de", "max", "maximum", "pas plus",
    "seulement", "avec juste", "moins cher", "jusqu'à", "jusqu a",
    # existence framing ("is there something for X DA")
    "est-ce que", "est ce que", "y a-t-il", "y a t il", "avez-vous", "existe",
    # Darija (romanized)
    "3andi", "andi", "nhab", "ndir", "3la",
    # Arabic script
    "عندي", "معي", "لدي", "أريد", "اريد", "بـ",
    # English
    "i have", "can i get", "with only", "under",
]

# ---------------------------------------------------------------------------
# Catalogue intent  ("list your offers")
# ---------------------------------------------------------------------------
CATALOGUE_TRIGGERS = [
    "vos offres", "vos forfaits", "quelles offres", "quelles sont", "catalogue",
    "liste", "toutes les offres", "tous les forfaits", "proposez", "disponible",
    "your offers", "what offers", "what plans", "list",
    "عروض", "العروض", "وش عندكم", "واش كاين",       # Arabic / Darija
]

# ---------------------------------------------------------------------------
# Comparison intent  ("difference between A and B")
# ---------------------------------------------------------------------------
COMPARISON_TRIGGERS = [
    "différence", "difference", "entre", "comparer", "comparaison",
    "versus", "vs", "ou bien", "plutôt", "mieux entre",
    "difference between", "compare", "or",
    "الفرق", "بين", "مقارنة",                        # Arabic
]

# ---------------------------------------------------------------------------
# Darija markers (for language detection: detect dz before falling back to fr)
# ---------------------------------------------------------------------------
DARIJA_WORDS = {
    "bghit", "nhab", "nchri", "bezaf", "barcha", "chhal", "kifash", "kayen",
    "wesh", "wech", "win", "3andi", "andi", "3tini", "3la", "eddir", "ndir",
    "khlass", "sahbi", "khoya", "mlih", "labas", "wahed", "zouj", "hadi",
    "hada", "ki", "rani", "raki", "wash", "chwiya", "daba",
    # Arabizi spellings using digits for Arabic letters (7=ح, 9=ق, 3=ع): the
    # "how much" word is commonly typed "ch7al"/"che7al"/"ch9al", which the plain
    # "chhal" entry missed -> the question was misdetected as French.
    "ch7al", "che7al", "ch9al", "ch7all", "kch7al",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Pre-compute, for single-word synonym keys, a word-boundary regex so that
# expanding "data" doesn't fire inside "database". Multi-word keys use substring.
_SINGLE_WORD_KEYS = {k: re.compile(rf"\b{re.escape(k)}\b") for k in SYNONYMS if " " not in k}


# --- accent-aware, word-boundary offer-name matching ----------------------
def _fold(s: str) -> str:
    """Lowercase, strip accents (é→e, ô→o), and treat - and _ as spaces.

    Hyphen/underscore folding makes URL-slug spellings ("flexy-net",
    "facebook-flex") match the spaced prose ("Flexy Net", "Facebook Flex"). The
    substitution is 1:1 so character offsets are preserved for span tracking.
    """
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return s.replace("-", " ").replace("_", " ")


# Some offers have surface variants written differently in pages vs queries.
_OFFER_VARIANTS = {"campuce": ["campuce", "cam puce"]}

# Arabic-script spellings of brand offer names → canonical Latin name. Arabic
# speakers sometimes transliterate the brand ("ليجند" for Legend) instead of writing
# it in Latin. We map only UNAMBIGUOUS transliterations to the canonical Latin name,
# so downstream chunk matching (the scraped pages keep the Latin brand) still works.
# Ambiguous ones are deliberately left out — e.g. "زيد" is also the everyday verb
# "to add", which would cause false matches just like bare "Zid"/"Zidane" in Latin.
OFFER_ALIASES = {
    "ليجند": "legend",
    "ليجاند": "legend",
    "ايزي": "izzy",
    "إيزي": "izzy",
    "كونفور": "confort",
}

# Cache one word-boundary regex per (accent-folded) surface form.
_OFFER_PATTERNS = {}


def _offer_pattern(form: str):
    pat = _OFFER_PATTERNS.get(form)
    if pat is None:
        pat = re.compile(rf"\b{re.escape(_fold(form))}\b")
        _OFFER_PATTERNS[form] = pat
    return pat


def offer_in_text(name: str, text: str) -> bool:
    """True if offer `name` appears in `text` as a whole word (accent-insensitive).

    Word boundaries stop short names matching inside other words ("control" won't
    match "contrôle", "zid" won't match "Zidane"), and accent folding lets a query
    match accented page text. Handles surface variants (campuce / cam puce).
    """
    folded = _fold(text)
    forms = _OFFER_VARIANTS.get(name, [name])
    return any(_offer_pattern(f).search(folded) for f in forms)


def expand_synonyms(query: str) -> str:
    """Append synonym expansions for any lexicon phrase found in `query`.

    Returns the original query plus space-joined expansion terms. Matching is
    case-insensitive; single-word keys match on word boundaries, multi-word keys
    match as substrings. Order of `SYNONYMS` is preserved for deterministic output.
    """
    q = query.lower()
    extras = []
    for phrase, expansion in SYNONYMS.items():
        if " " in phrase:
            if phrase in q:
                extras.append(expansion)
        else:
            if _SINGLE_WORD_KEYS[phrase].search(q):
                extras.append(expansion)
    if not extras:
        return query
    return query + " " + " ".join(extras)


# Words that signal a telecom / Djezzy / price intent. Used by the out-of-domain
# guard: a query with NONE of these (and no offer name) is treated as off-topic.
# Generous on purpose — includes price/purchase intent so short follow-ups like
# "c'est combien ?" stay in-domain. Latin entries match on word boundaries;
# Arabic entries match as substrings.
TELECOM_SIGNALS = [
    # core telecom nouns
    "forfait", "forfaits", "offre", "offres", "internet", "data", "go", "mo",
    "da", "dinar", "dinars", "dzd", "sms", "appel", "appels", "minute", "minutes",
    "roaming", "djezzy", "5g", "4g", "3g", "credit", "recharge", "giga", "gigas",
    "social", "facebook", "instagram", "tiktok", "youtube", "mobile", "puce",
    "sim", "numero", "ussd", "tarif", "tarifs", "abonnement", "reseau", "reseaux",
    "connexion", "debit", "volume", "flexy", "modem", "wifi", "ligne", "validite",
    "mega", "telephone", "smartphone", "portable", "appareil", "service",
    # price / purchase intent (keeps short follow-ups in domain)
    "combien", "prix", "coute", "coutent", "cout", "payer", "acheter", "souscrire",
    "cher", "chhal",
    # Arabic — offers/price/telecom nouns + DEVICES (phones), so Arabic questions
    # like "هل تبيعون هواتف؟" reach retrieval instead of the out-of-domain refusal.
    "عرض", "عروض", "انترنت", "رصيد", "سعر", "اسعار", "مكالمات", "باقة", "جيجا", "روم",
    "هاتف", "هواتف", "تليفون", "تيليفون", "جوال", "موبايل", "فون", "شريحة", "خط",
]

_TELECOM_ASCII_RE = [
    (s, re.compile(rf"\b{re.escape(s)}\b")) for s in TELECOM_SIGNALS if s.isascii()
]
_TELECOM_NONASCII = [s for s in TELECOM_SIGNALS if not s.isascii()]


def has_telecom_signal(query: str) -> bool:
    """True if `query` carries any telecom/price/offer/device/service intent."""
    if detect_offers(query):
        return True
    folded = _fold(query)
    if any(rx.search(folded) for _, rx in _TELECOM_ASCII_RE):
        return True
    if any(s in query for s in _TELECOM_NONASCII):
        return True
    # device (phones) and service content types
    return any(_fold(t) in folded for t in (SERVICE_TERMS + DEVICE_TERMS))


def detect_offers(query: str) -> list:
    """Return the canonical offer names mentioned in `query`, most-specific first.

    Used by the catalogue / named-offer / comparison routes. De-duplicates and
    avoids double-counting overlapping names (e.g. won't report both
    "legend max" and "legend" for the text "legend max").
    """
    folded = _fold(query)
    found = []
    consumed_spans = []
    for name in OFFER_NAMES:  # OFFER_NAMES is ordered most-specific first
        m = _offer_pattern(name).search(folded)   # whole-word, accent-insensitive
        if not m:
            continue
        span = m.span()
        # skip if this match sits inside an already-matched (longer) name
        if any(s <= span[0] and span[1] <= e for s, e in consumed_spans):
            continue
        # normalize the two spellings of Cam Puce to a single canonical token
        canonical = "campuce" if name in ("cam puce", "campuce") else name
        if canonical not in found:
            found.append(canonical)
        consumed_spans.append(span)
    # Arabic-script brand spellings → canonical Latin name (so an Arabic query like
    # "عرض ليجند" is detected as the Legend offer and matched against the Latin pages).
    for ar, canon in OFFER_ALIASES.items():
        if ar in folded and canon not in found:
            found.append(canon)
    return found
