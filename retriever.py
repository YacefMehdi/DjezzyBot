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

# Budget amount: a number followed by a dinar token. Covers the Latin forms
# (da / dinars / dzd), the abbreviated Arabic (دج / د.ج) AND the full Arabic word
# دينار / دنانير — Arabic speakers commonly write the amount as "1000 دينار", which
# the abbreviation-only pattern used to miss (the query then misrouted off budget).
_BUDGET_RE = re.compile(
    r"(\d[\d\s]*)\s*(?:da|dinars?|dzd|دينار|دنانير|دج|د\.?ج)\b", re.IGNORECASE)

# Price inside a chunk, e.g. "3 000 DA", "500 DA".
_PRICE_RE = re.compile(r"(\d[\d\s]{0,7}\d|\d)\s*(?:da|dinars?|dzd)\b", re.IGNORECASE)

# A DA amount immediately followed by "/ <unit>" is a per-UNIT RATE ("5 DA/SMS",
# "4.99 DA / Mo", "5 DA/30 Sec"), NOT a subscription price. Rates must never be read
# as an offer's price — they were polluting the budget route with tariff-table noise
# (every "5 DA/SMS" looked like a 5 DA offer).
_RATE_AFTER_RE = re.compile(
    r"\s*/\s*\d*\s*(?:sms|mo|mb|go|ko|min|sec|secondes?|message|appel)", re.IGNORECASE)

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


def _copy_doc(doc, new_text: str = None):
    """A copy of a Document (optionally with replaced text).

    Retrieval must NEVER mutate the chunks stored in the FAISS docstore — they are
    reused by every later query. Routes that trim or rewrite chunk text (the budget
    filter, catalogue snippets) work on copies returned by this helper.
    """
    from langchain_core.documents import Document
    return Document(
        page_content=doc.page_content if new_text is None else new_text,
        metadata=dict(doc.metadata),
    )


def _offer_page_url(store: dict, offer: str):
    """URL of the page that best represents `offer`, or None.

    Prefers the page whose URL slug contains the offer name (its dedicated page);
    otherwise the page with the most name-matching chunks.
    """
    folded = lexicon._fold(offer).replace(" ", "")
    counts = {}
    slug_url = None
    for d in store.values():
        if lexicon.offer_in_text(offer, d.page_content):
            url = d.metadata.get("source_url", "")
            counts[url] = counts.get(url, 0) + 1
            if slug_url is None and folded and \
                    folded in lexicon._fold(url).replace(" ", "").replace("/", ""):
                slug_url = url
    if slug_url:
        return slug_url
    return max(counts, key=counts.get) if counts else None


def _page_chunks_for_offer(vector_db, offer: str, limit: int = None):
    """All chunks of the offer's OWN page, in page order (so every tier is present).

    The completeness fix: a named offer is answered from its whole page, not only
    the chunks that repeat its name — later chunks holding more forfait tiers (which
    don't restate the name) are no longer dropped. Falls back to name-matching
    chunks if the page can't be identified.
    """
    try:
        store = vector_db.docstore._dict
    except AttributeError:
        return []
    url = _offer_page_url(store, offer)
    if url:
        docs = [d for d in store.values() if d.metadata.get("source_url", "") == url]
    else:
        docs = [d for d in store.values() if lexicon.offer_in_text(offer, d.page_content)]
    return docs[:limit] if limit is not None else docs


def _snippet(doc, max_chars: int = 480):
    """A compact COPY of a catalogue chunk (name + starting price fit in ~480 chars).

    Catalogue answers are a brief one-line-per-gamme menu, so the context only needs
    a short head of each offer's chunk — this keeps every gamme inside the context
    window instead of the cap truncating the list to the first few.
    """
    return _copy_doc(doc, doc.page_content[:max_chars])


def _catalogue_snippet(vector_db, offer: str):
    """For the catalogue menu: a snippet that shows this gamme's STARTING price,
    plus that price (for cheapest-first sorting).

    Picks the cheapest PRICED chunk that names the offer, so the catalogue line
    always carries an "à partir de X DA" (the old one-chunk-per-name pick often
    landed on a chunk with no price, so the model silently dropped the gamme).
    Falls back to any name chunk. Returns (snippet_doc | None, price | None).
    """
    try:
        store = vector_db.docstore._dict
    except AttributeError:
        return None, None
    named = [d for d in store.values() if lexicon.offer_in_text(offer, d.page_content)]
    if not named:
        return None, None
    priced = [(_chunk_price(d.page_content), d) for d in named
              if _chunk_price(d.page_content) is not None]
    if priced:
        price, doc = min(priced, key=lambda x: x[0])
        return _snippet(doc), price
    return _snippet(named[0]), None


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
    """Extract the smallest DA SUBSCRIPTION price in a chunk, or None.

    Per-unit rates ("5 DA/SMS", "4.99 DA/Mo") are skipped — they are tariffs, not
    prices, and would otherwise make every tariff table look like a cheap offer."""
    low = text.lower()
    prices = []
    for m in _PRICE_RE.finditer(low):
        if _RATE_AFTER_RE.match(low, m.end()):
            continue                            # per-unit rate, not a price
        try:
            prices.append(int(m.group(1).replace(" ", "")))
        except ValueError:
            continue
    return min(prices) if prices else None


def _names_an_offer(text: str) -> bool:
    """True if the chunk mentions a known offer name — used to drop nameless
    fee/tariff tables from the budget pool (they caused hallucinated offer names)."""
    return any(lexicon.offer_in_text(n, text) for n in lexicon.OFFER_NAMES)


def _doc_price(doc):
    """A chunk's sort key: its cheapest tier price, or +inf if it carries none.

    Used to order priced chunks/offers cheapest-first in Python (so the answer's
    order never depends on the LLM honouring 'du moins cher au plus cher')."""
    p = _chunk_price(doc.page_content)
    return p if p is not None else float("inf")


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
    if _RATE_AFTER_RE.match(lines[i].lower(), m.end()):
        return None                            # per-unit rate (5 DA/SMS), not a tier
    ctx = " ".join(lines[i: i + 2]).lower()
    if any(w in ctx for w in _CREDIT_WORDS):
        return None
    try:
        return int(m.group(1).replace(" ", ""))
    except ValueError:
        return None


def _split_offer_blocks(text: str):
    """Split offer text into [price_or_None, [lines]] blocks at tier-price boundaries.

    Block 0 is the leading preamble/header (price None, possibly empty); each later
    block starts at a tier (subscription) price line and runs until the next one, so
    a tier's price stays glued to its own details. Defined ONCE here and shared by
    the budget filter and the price-sorter, so "what counts as a tier boundary" can
    never drift between the two.
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
    return blocks


def _sort_offer_text_by_price(text: str) -> str:
    """Reorder an offer chunk's tier blocks cheapest-first, deterministically.

    Keeps the leading preamble in place, then orders the priced tiers ascending
    (Cam Puce's 7 paliers, Legend's tiers...). This is the system-level fix for the
    "random tier order" bug: the context is ALREADY cheapest-first, so the answer's
    order no longer depends on the LLM obeying 'liste du moins cher au plus cher'.
    A chunk with 0–1 tiers is returned unchanged (nothing to reorder).
    """
    blocks = _split_offer_blocks(text)
    if len(blocks) <= 2:                         # preamble + at most one tier
        return text
    preamble = "\n".join(blocks[0][1])           # block 0 = header (price None)
    priced = sorted((( p, "\n".join(blk)) for p, blk in blocks[1:]),
                    key=lambda x: x[0])
    ordered = ([preamble] if preamble.strip() else []) + [t for _, t in priced]
    return "\n".join(t for t in ordered if t.strip())


def _redact_over_budget(text: str, budget: int) -> str:
    """Drop any LINE that carries a DA amount strictly greater than `budget`.

    Even inside an AFFORDABLE offer block, a crédit/bonus figure or a preamble teaser
    can name a sum above the budget ("2 000 DA de crédit"), and the model then re-prints
    it as if it were a purchasable price (the r17 leak: over_in_context=[2000] for a
    600 DA budget). For a budget answer the client only cares about affordable figures,
    so we remove the whole offending line. We do NOT substitute placeholder text:
    an earlier version inserted a French phrase, which destabilised Qwen on Arabic
    budget prompts (it code-switched to Chinese — the r01 regression). Dropping the
    line keeps the context in its original language. Lines whose only DA amounts are
    <=budget (the real tier prices) are kept untouched.
    """
    out = []
    for line in text.split("\n"):
        over = False
        for m in _PRICE_RE.finditer(line):
            try:
                if int(m.group(1).replace(" ", "")) > budget:
                    over = True
                    break
            except ValueError:
                continue
        if not over:
            out.append(line)
    return "\n".join(out)


def _filter_offer_text_by_budget(text: str, budget: int):
    """Return (filtered_text, cheapest_kept_price) keeping only <=budget offers.

    Splits the chunk into offer blocks at tier-price boundaries, keeps the leading
    preamble plus every block whose tier price is <= budget, and drops blocks
    priced above budget. Crédit/bonus DA amounts stay inside their block (they're
    not boundaries) but any amount ABOVE the budget is then redacted, so an affordable
    offer keeps its full details without leaking an over-budget figure into the answer.
    Returns cheapest_kept_price = None when the chunk has no affordable tier at all.
    """
    blocks = _split_offer_blocks(text)
    preamble, kept = [], []                         # kept: (price, block_text)
    for idx, (price, blk) in enumerate(blocks):
        if price is None:
            if idx == 0:                            # leading preamble / header
                preamble.append("\n".join(blk))
        elif price <= budget:
            kept.append((price, "\n".join(blk)))
    if not kept:
        return "", None
    kept.sort(key=lambda x: x[0])                   # cheapest-first, like every route
    ordered = preamble + [t for _, t in kept]
    text_out = "\n".join(t for t in ordered if t.strip())
    return _redact_over_budget(text_out, budget), kept[0][0]


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
        # emit a COPY with only the affordable blocks — never mutate the docstore
        # chunk (it must stay intact for later queries)
        out.append((cheapest, _copy_doc(d, filtered)))
    out.sort(key=lambda x: x[0])            # cheapest first
    return [doc for _, doc in out[: config.K_BUDGET_RETURN]]


def classify_route(query: str) -> str:
    """The single source of truth for the user's INTENT (the route).

    smart_retrieve dispatches on this, and bot.py reads it to choose the answer
    STYLE and the budget ceiling — so retrieval scope and presentation can never
    drift apart (that disconnect was the cause of the "dumps everything" and
    "random order" bugs). Precedence, highest first:

        competitor > roaming > catalogue > comparison > budget > named > normal

    Notes: catalogue yields to budget (an explicit amount means "filter", not "list
    all"); a 2-offer comparison wins over budget. The NORMAL route may still resolve
    to an out-of-domain refusal inside smart_retrieve, which needs retrieval
    confidence a pure classifier can't see.
    """
    if _is_competitor_query(query):
        return "competitor"
    offers = lexicon.detect_offers(query)
    if _roaming_markers(query):
        return "roaming"
    if _is_catalogue_query(query, offers) and not _is_budget_query(query):
        return "catalogue"
    if _is_comparison_query(query, offers):
        return "comparison"
    if _is_budget_query(query):
        return "budget"
    if offers:
        return "named"
    return "normal"


def budget_of(query: str):
    """Public: the budget amount in DA if `query` is BUDGET-routed, else None.

    Lets bot.py inject the exact budget value into the prompt so the LLM also
    enforces the ceiling (belt-and-suspenders with the Python filter). Non-None only
    when classify_route says budget, so the prompt's "context is pre-filtered" claim
    is always truthful (roaming/comparison queries that mention an amount get None).
    """
    return _extract_budget(query) if classify_route(query) == "budget" else None


# ===========================================================================
# Public entry point
# ===========================================================================
def smart_retrieve(query: str, lang: str, vector_db):
    """Route `query` to the right retrieval strategy and return chunks.

    Dispatches on classify_route() — the single intent classifier shared with
    bot.py — so retrieval scope and the answer style chosen later can never drift
    apart. Returns config.COMPETITOR_SENTINEL for a competitor question, otherwise
    a list of LangChain Documents (possibly empty when the OOD guard rejects a
    normal query).
    """
    route = classify_route(query)

    # --- COMPETITOR firewall ----------------------------------------------
    if route == "competitor":
        logger.info("route=competitor")
        return config.COMPETITOR_SENTINEL

    expanded = lexicon.expand_synonyms(query)   # richer retrieval string
    offers = lexicon.detect_offers(query)       # canonical names mentioned

    # --- ROAMING (wins over budget: a foreign price is never filtered national) --
    if route == "roaming":
        roaming = _roaming_markers(query)
        logger.info("route=roaming markers=%s", roaming)
        docs = _faiss(vector_db, expanded, config.K_ROAMING * 2)
        preferred, other = [], []
        for d in docs:
            blob = (d.metadata.get("source_url", "") + " " + d.page_content).lower()
            if "roaming" in blob or any(m in blob for m in roaming):
                preferred.append(d)
            else:
                other.append(d)
        return _dedup(preferred + other)[: config.K_ROAMING]

    # --- CATALOGUE: one PRICED snippet per gamme, sorted cheapest-first ----------
    # Every distinct gamme gets a snippet that carries its starting price, and the
    # whole list is ordered in Python (the model won't sort reliably). This fixes
    # both the missing-gamme and random-order problems at the source.
    if route == "catalogue":
        logger.info("route=catalogue")
        entries, seen, seen_urls = [], set(), set()
        for name in lexicon.OFFER_NAMES:
            canonical = "campuce" if name in ("cam puce", "campuce") else name
            if canonical in seen:
                continue
            doc, price = _catalogue_snippet(vector_db, name)
            if doc is None:
                continue
            url = doc.metadata.get("source_url", "")
            if url and url in seen_urls:        # overlap (e.g. "legend" hit a Legend
                continue                        # Max page already listed) → skip
            seen.add(canonical)
            seen_urls.add(url)
            entries.append((price if price is not None else float("inf"), doc))
        if not entries:  # nothing indexed under those names → fall back to dense
            return _faiss(vector_db, expanded, config.K_NORMAL)
        entries.sort(key=lambda x: x[0])        # cheapest gamme first, priceless last
        return [d for _, d in entries]

    # --- COMPARISON: each named offer's page (capped), both guaranteed -----------
    # Offers presented cheapest-first, and each offer's own tiers sorted too — same
    # deterministic Python ordering as catalogue/budget, never left to the LLM.
    if route == "comparison":
        logger.info("route=comparison offers=%s", offers)
        groups = []                              # (cheapest_price, [chunks]) per offer
        for name in offers:
            chunks = _page_chunks_for_offer(vector_db, name, config.K_COMPARISON_PER_OFFER)
            if not chunks:  # ensure presence even if the page can't be identified
                chunks = _faiss(vector_db, name, config.K_COMPARISON_PER_OFFER)
            chunks = [_copy_doc(d, _sort_offer_text_by_price(d.page_content)) for d in chunks]
            groups.append((min((_doc_price(d) for d in chunks), default=float("inf")),
                           chunks))
        groups.sort(key=lambda g: g[0])          # cheapest offer first
        return _dedup([d for _, chunks in groups for d in chunks])

    # --- BUDGET: pool + Python price filter (cheapest first, LLM never filters) --
    if route == "budget":
        budget = _extract_budget(query)
        logger.info("route=budget budget=%s DA", budget)
        pool = _faiss(vector_db, expanded, config.K_BUDGET_POOL)
        # also pull every offer's chunks so cheap offers aren't missed by dense rank
        for name in lexicon.OFFER_NAMES:
            pool.extend(_exact_match_chunks(vector_db, name, 2))
        pool = _dedup(pool)
        # keep only chunks that NAME an offer — drops generic fee/tariff tables that
        # have no offer and led the model to invent offer names. Fall back to the full
        # pool if nothing names an offer (so budget never returns empty by accident).
        named = [d for d in pool if _names_an_offer(d.page_content)]
        return _filter_by_budget_python(named or pool, budget)

    # --- NAMED OFFER: the offer's WHOLE page (all tiers) + a couple dense --------
    if route == "named":
        logger.info("route=named_offer offers=%s", offers)
        primary = offers[0]
        page = _page_chunks_for_offer(vector_db, primary)        # whole page, all tiers
        if page:
            # Deterministic cheapest-first ordering in Python: tiers WITHIN each chunk
            # and chunks AMONG themselves. This is the Cam Puce fix (7 paliers were
            # returned in arbitrary page order) — generalised to every named offer.
            page = [_copy_doc(d, _sort_offer_text_by_price(d.page_content)) for d in page]
            page.sort(key=_doc_price)                            # cheapest chunk first
            docs = page
        else:  # offer not located by page → fall back to name-match chunks
            docs = _exact_match_chunks(vector_db, primary, config.K_NAMED_EXACT)
        docs = docs + _faiss(vector_db, expanded, config.K_NAMED_FAISS)
        return _dedup(docs)

    # --- NORMAL: dense search, guarded against out-of-domain questions -----------
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
    """True if `query` is off-topic → the bot gives the no-context refusal.

    CONFIDENCE-based, not a keyword list:
      * A clear telecom/offer signal is a fast-pass → in-domain, answered whatever
        the score. (The keyword list now only ever HELPS; it can never wrongly
        refuse, which is what made phones-in-Arabic fail before.)
      * Without a signal we DON'T refuse on the missing keyword. We judge by
        RETRIEVAL CONFIDENCE: off-topic only when the best chunk scores below
        OOD_MIN_SIMILARITY. Self-adapting (no vocabulary to maintain) and still
        decided before the LLM runs, so off-domain stays a fast refusal.

    Scores come from similarity_search_with_score on a MAX_INNER_PRODUCT index, so
    HIGHER = more similar and scored_docs[0] is the best match. OOD_MIN_SIMILARITY
    MUST be calibrated on the real index (e5 scores cluster high) — see config.py.
    """
    if lexicon.has_telecom_signal(query):
        return False                          # clear in-domain signal → answer
    if not scored_docs:
        return False                          # no score to judge on → let it through
    return scored_docs[0][1] < config.OOD_MIN_SIMILARITY
