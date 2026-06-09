"""
retriever.py — smart_retrieve(): the intent router in front of FAISS.

This is where the old bot's bugs lived, so the routing is explicit and ordered.
Given a user query, we decide WHICH retrieval strategy to use before touching the
vector store, then return the chunks (or the COMPETITOR sentinel).

    smart_retrieve(query, lang, vector_db) -> list[Document] | "COMPETITOR"

Pre-processing (always):
    1. competitor check  -> may short-circuit to the sentinel
    2. synonym expansion (lexicon)         -> richer retrieval string
    3. detect mentioned offers / budget / roaming / catalogue / comparison intents

Modes, in PRIORITY ORDER:
    COMPETITOR  -> return "COMPETITOR"
    ROAMING     -> k=6, prefer chunks whose URL/text carries a roaming marker
    CATALOGUE   -> 1 exact-match chunk per known offer, return all
    COMPARISON  -> 4 chunks per named offer, both offers guaranteed present
    BUDGET      -> pull k=20 price chunks, FILTER PRICES IN PYTHON (<= budget),
                   return up to 5. The LLM never does the filtering.
    NAMED OFFER -> 3 exact-match + 2 FAISS
    NORMAL      -> k=5

Every FAISS call receives the RAW (un-prefixed) string; the indexer's embedding
adapter adds the "query: " prefix. We never prefix here (that would double it).

Note on "exact-match chunk": a lightweight lexical match over the indexed chunks
(offer name appears in the chunk text), used to guarantee the right offer is
present even when dense retrieval would rank it lower.
"""

import re
import logging

import config
from data import lexicon

logger = logging.getLogger("djezzybot.retriever")

# Arabic-Indic digits -> ASCII, so "عندي ٥٠٠ دج" parses like "500 da".
_ARAB_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# Budget amount: a number followed by a dinar token (da / dinars / دج / د.ج).
_BUDGET_RE = re.compile(r"(\d[\d\s]*)\s*(?:da|dinars?|dzd|دج|د\.?ج)\b", re.IGNORECASE)

# Price inside a chunk, e.g. "3 000 DA", "500 DA".
_PRICE_RE = re.compile(r"(\d[\d\s]{0,7}\d|\d)\s*(?:da|dinars?|dzd)\b", re.IGNORECASE)

# Words that mark a DA amount as a crédit / bonus / gift rather than the offer's
# subscription PRICE. A Djezzy tier like "1 200 DA" can *include* "3 000 DA de
# crédit" — that 3000 is not a price and must not be treated as a tier boundary
# or as something that breaks the budget ceiling.
_CREDIT_WORDS = ("credit", "crédit", "crèdit", "recharge", "bonus", "cadeau", "offert")


# ===========================================================================
# Intent detection (pure helpers)
# ===========================================================================
def _extract_budget(query: str):
    """Return the budget amount in DA as int, or None if no amount present."""
    q = query.lower().translate(_ARAB_DIGITS)
    m = _BUDGET_RE.search(q)
    if not m:
        return None
    try:
        return int(m.group(1).replace(" ", ""))
    except ValueError:
        return None


def _is_budget_query(query: str) -> bool:
    """A budget query needs BOTH a parsed amount AND a budget intent cue."""
    if _extract_budget(query) is None:
        return False
    q = query.lower().translate(_ARAB_DIGITS)
    return any(cue in q for cue in lexicon.BUDGET_TRIGGERS)


def _roaming_markers(query: str):
    """Return roaming destination markers if this is a roaming query, else None."""
    q = query.lower()
    markers = []
    for dest, marks in lexicon.ROAMING_TRIGGERS.items():
        if dest in q:
            markers.extend(marks)
    if markers:
        return list(dict.fromkeys(markers))  # dedup, keep order
    # generic "roaming/étranger" + a travel cue
    if any(g in q for g in lexicon.ROAMING_GENERIC) and \
            any(c in q for c in lexicon.ROAMING_TRAVEL_CUES):
        return ["roaming"]
    # bare "roaming" with no destination still counts as roaming intent
    if "roaming" in q:
        return ["roaming"]
    return None


def _is_catalogue_query(query: str, offers: list) -> bool:
    """True for 'list your offers' questions that don't name a specific offer."""
    q = query.lower()
    return any(t in q for t in lexicon.CATALOGUE_TRIGGERS) and not offers


def _is_comparison_query(query: str, offers: list) -> bool:
    """True when a comparison cue is present AND >= 2 distinct offers are named."""
    q = query.lower()
    return any(t in q for t in lexicon.COMPARISON_TRIGGERS) and len(offers) >= 2


def _is_competitor_query(query: str) -> bool:
    """True if `query` is really ABOUT a competitor (→ firewall refusal).

    A bare brand mention is NOT enough: Djezzy offers say "appels vers Ooredoo /
    Mobilis", so "est-ce que iZZY permet d'appeler Mobilis ?" must be answered,
    not refused. We refuse when:
      * a generic competition word appears ("concurrent", "competitor", ...), or
      * a competitor brand appears WITHOUT a call-destination cue before it
        ("offres Ooredoo", "Mobilis propose", "Ooredoo vs Djezzy").
    We allow (route normally) when every brand mention is preceded by a call
    cue ("vers/appeler/calls to Mobilis").
    """
    q = lexicon._fold(query)
    if any(g in q for g in lexicon.COMPETITOR_GENERIC):
        return True
    brands = [b for b in lexicon.COMPETITOR_BRANDS if lexicon._fold(b) in q]
    if not brands:
        return False
    for b in brands:
        i = q.find(lexicon._fold(b))
        before = q[max(0, i - 30): i]
        if not any(cue in before for cue in lexicon.CALL_DEST_CUES):
            return True          # this brand is the subject → refuse
    return False                 # every brand was a call destination → allow


# ===========================================================================
# Chunk-level helpers
# ===========================================================================
def _exact_match_chunks(vector_db, offer: str, limit: int):
    """Return up to `limit` indexed chunks whose text contains `offer` (lexical).

    Walks the FAISS docstore directly (no embedding) so the named offer is
    guaranteed present even if dense retrieval would have missed it. Uses
    whole-word, accent-insensitive matching (lexicon.offer_in_text) so a short
    name like "zid" or "control" doesn't match inside an unrelated word.
    """
    out = []
    try:
        store = vector_db.docstore._dict  # LangChain InMemoryDocstore
    except AttributeError:
        return out
    for doc in store.values():
        if lexicon.offer_in_text(offer, doc.page_content):
            out.append(doc)
            if len(out) >= limit:
                break
    return out


def _faiss(vector_db, query: str, k: int):
    """similarity_search with the RAW query (adapter adds the 'query:' prefix)."""
    return vector_db.similarity_search(query, k=k)


def _dedup(docs: list) -> list:
    """Remove duplicate chunks (same source_url + same text) preserving order."""
    seen = set()
    out = []
    for d in docs:
        key = (d.metadata.get("source_url", ""), d.page_content)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _chunk_price(text: str):
    """Extract the smallest DA price found in a chunk, or None."""
    prices = []
    for m in _PRICE_RE.finditer(text.lower()):
        try:
            prices.append(int(m.group(1).replace(" ", "")))
        except ValueError:
            continue
    return min(prices) if prices else None


def _is_tier_price(lines: list, i: int):
    """If line `i` holds a tier (subscription) price, return it, else None.

    A DA amount is a tier price unless a crédit/bonus word sits on the SAME line
    or the NEXT line — Djezzy renders the amount then its label ("3 000 DA" then
    "CRÉDIT"). We deliberately do NOT look at the previous line: a crédit label
    belonging to the prior offer must not suppress the next offer's real price.
    """
    m = _PRICE_RE.search(lines[i])
    if not m:
        return None
    ctx = " ".join(lines[i: i + 2]).lower()
    if any(w in ctx for w in _CREDIT_WORDS):
        return None
    try:
        return int(m.group(1).replace(" ", ""))
    except ValueError:
        return None


def _filter_offer_text_by_budget(text: str, budget: int):
    """Return (filtered_text, cheapest_kept_price) keeping only <=budget offers.

    Splits the chunk into offer blocks at tier-price boundaries, keeps the leading
    preamble plus every block whose tier price is <= budget, and drops blocks
    priced above budget. Crédit/bonus DA amounts stay inside their block (they're
    not boundaries), so an affordable offer keeps its full details. Returns
    cheapest_kept_price = None when the chunk has no affordable tier at all.
    """
    lines = text.split("\n")
    blocks = []                 # list of [price_or_None, [lines]]
    cur_price, cur_lines = None, []
    for i, line in enumerate(lines):
        price = _is_tier_price(lines, i)
        if price is not None:
            blocks.append([cur_price, cur_lines])   # close previous block
            cur_price, cur_lines = price, [line]
        else:
            cur_lines.append(line)
    blocks.append([cur_price, cur_lines])

    kept_text, kept_prices = [], []
    for idx, (price, blk) in enumerate(blocks):
        if price is None:
            if idx == 0:                            # leading preamble / header
                kept_text.append("\n".join(blk))
        elif price <= budget:
            kept_text.append("\n".join(blk))
            kept_prices.append(price)
    if not kept_prices:
        return "", None
    return "\n".join(t for t in kept_text if t.strip()), min(kept_prices)


def _filter_by_budget_python(docs: list, budget: int) -> list:
    """Budget filter at OFFER-BLOCK granularity, done in Python (never the LLM).

    For each chunk, drop the offer blocks priced above budget and keep only the
    affordable ones (with their crédit/bonus details intact). A chunk with no
    affordable tier is dropped entirely. Remaining chunks are returned cheapest
    first so the most affordable options survive the K_BUDGET_RETURN cap.

    This fixes the granularity bug: a 1200-char chunk holding both a 300 DA and a
    2000 DA offer no longer leaks the 2000 DA offer into the context just because
    its cheapest price passed the filter.
    """
    out = []
    for d in docs:
        filtered, cheapest = _filter_offer_text_by_budget(d.page_content, budget)
        if cheapest is None:
            continue
        d.page_content = filtered           # keep only the affordable blocks
        out.append((cheapest, d))
    out.sort(key=lambda x: x[0])            # cheapest first
    return [doc for _, doc in out[: config.K_BUDGET_RETURN]]


def _budget_route_active(query: str) -> bool:
    """True iff smart_retrieve will actually take the BUDGET route for `query`.

    Mirrors smart_retrieve's precedence (competitor > roaming > catalogue >
    comparison > budget): budget is the live route only when it's a budget query
    AND no higher-priority route claims it. This keeps bot.py's budget ceiling in
    the prompt consistent with the context it was actually given — a roaming or
    comparison query that happens to mention an amount must NOT get a budget note,
    because its context was never Python-filtered to that amount.
    """
    if not _is_budget_query(query):
        return False
    if _is_competitor_query(query):
        return False
    if _roaming_markers(query):              # roaming wins over budget (by design)
        return False
    offers = lexicon.detect_offers(query)
    # catalogue YIELDS to budget (an explicit amount means "filter", not "list all"),
    # so it does not block here; a 2-offer comparison still wins over budget.
    if _is_comparison_query(query, offers):
        return False
    return True


def budget_of(query: str):
    """Public: the budget amount in DA if `query` is BUDGET-routed, else None.

    Lets bot.py inject the exact budget value into the prompt so the LLM also
    enforces the ceiling (belt-and-suspenders with the Python filter above). Only
    returns a value when the budget route is the one smart_retrieve actually runs,
    so the prompt's "context is pre-filtered" claim is always truthful.
    """
    return _extract_budget(query) if _budget_route_active(query) else None


# ===========================================================================
# Public entry point
# ===========================================================================
def smart_retrieve(query: str, lang: str, vector_db):
    """Route `query` to the right retrieval strategy and return chunks.

    Returns the string sentinel config.COMPETITOR_SENTINEL ("COMPETITOR") if the
    query mentions a competitor; otherwise a list of LangChain Documents (possibly
    empty if nothing matched).
    """
    # --- 1. COMPETITOR firewall (highest priority) -------------------------
    # Only refuse when the query is really ABOUT a competitor — not when a brand
    # appears as a call destination inside a Djezzy offer ("appels vers Mobilis").
    if _is_competitor_query(query):
        logger.info("route=competitor")
        return config.COMPETITOR_SENTINEL

    # --- pre-processing ----------------------------------------------------
    expanded = lexicon.expand_synonyms(query)   # richer retrieval string
    offers = lexicon.detect_offers(query)       # canonical names mentioned
    budget = _extract_budget(query)
    roaming = _roaming_markers(query)

    # --- 2. ROAMING --------------------------------------------------------
    # Roaming wins over budget: a price abroad must never be filtered as national.
    if roaming:
        logger.info("route=roaming markers=%s", roaming)
        docs = _faiss(vector_db, expanded, config.K_ROAMING * 2)
        # prefer chunks whose URL or text actually carries a roaming marker
        preferred, other = [], []
        for d in docs:
            blob = (d.metadata.get("source_url", "") + " " + d.page_content).lower()
            if "roaming" in blob or any(m in blob for m in roaming):
                preferred.append(d)
            else:
                other.append(d)
        ranked = _dedup(preferred + other)[: config.K_ROAMING]
        return ranked

    # --- 3. CATALOGUE ------------------------------------------------------
    # "show me your offers" with no specific offer named: one chunk per known
    # offer so every gamme is represented. A catalogue phrasing that ALSO states a
    # budget ("j'ai 500 DA, que proposez-vous ?") yields to the BUDGET route below
    # so the context is actually price-filtered (never list the full catalogue and
    # then claim it was filtered).
    if _is_catalogue_query(query, offers) and not _is_budget_query(query):
        logger.info("route=catalogue")
        docs = []
        for name in lexicon.OFFER_NAMES:
            docs.extend(_exact_match_chunks(vector_db, name, config.K_CATALOGUE_PER_OFFER))
        docs = _dedup(docs)
        if not docs:  # nothing indexed under those names → fall back to dense
            docs = _faiss(vector_db, expanded, config.K_NORMAL)
        return docs

    # --- 4. COMPARISON -----------------------------------------------------
    # ">= 2 offers + a comparison cue": guarantee both offers are present.
    if _is_comparison_query(query, offers):
        logger.info("route=comparison offers=%s", offers)
        docs = []
        for name in offers:
            chunks = _exact_match_chunks(vector_db, name, config.K_COMPARISON_PER_OFFER)
            if not chunks:  # ensure presence even if lexical match is thin
                chunks = _faiss(vector_db, name, config.K_COMPARISON_PER_OFFER)
            docs.extend(chunks)
        return _dedup(docs)

    # --- 5. BUDGET ---------------------------------------------------------
    if _is_budget_query(query):
        logger.info("route=budget budget=%s DA", budget)
        pool = _faiss(vector_db, expanded, config.K_BUDGET_POOL)
        # also pull every offer's chunks so cheap offers aren't missed by dense rank
        for name in lexicon.OFFER_NAMES:
            pool.extend(_exact_match_chunks(vector_db, name, 2))
        pool = _dedup(pool)
        return _filter_by_budget_python(pool, budget)

    # --- 6. NAMED OFFER ----------------------------------------------------
    if offers:
        logger.info("route=named_offer offers=%s", offers)
        docs = []
        primary = offers[0]
        docs.extend(_exact_match_chunks(vector_db, primary, config.K_NAMED_EXACT))
        docs.extend(_faiss(vector_db, expanded, config.K_NAMED_FAISS))
        return _dedup(docs)

    # --- 7. NORMAL ---------------------------------------------------------
    # Fallback dense search, guarded against out-of-domain questions. Try a
    # scored search so the OOD backstop can see retrieval confidence; fall back
    # to plain search for stores that don't expose scores.
    try:
        scored = vector_db.similarity_search_with_score(expanded, k=config.K_NORMAL)
        docs = [d for d, _ in scored]
    except (AttributeError, TypeError):
        scored = None
        docs = _faiss(vector_db, expanded, config.K_NORMAL)

    if _is_out_of_domain(query, scored):
        logger.info("route=out_of_domain")
        return []
    logger.info("route=normal")
    return docs


def _is_out_of_domain(query: str, scored_docs) -> bool:
    """True if `query` is off-topic and the bot should give the no-context refusal.

    Primary gate (deterministic): the query carries no telecom/price/offer signal
    at all (e.g. "quelle est la météo") — those are out of domain regardless of
    what FAISS returned. Secondary backstop: even a signal-bearing query whose
    best chunk scores below OOD_MIN_SIMILARITY is treated as off-topic (only used
    when scores are available). Note we intentionally keep signal-bearing
    follow-ups like "c'est combien ?" in domain (price words are signals).
    """
    if not lexicon.has_telecom_signal(query):
        return True
    if scored_docs and scored_docs[0][1] < config.OOD_MIN_SIMILARITY:
        return True
    return False
