# DjezzyBot — Documentation & Performance Report

> **Status of the numbers in this report.** The architecture, data flow, and known
> limitations below are final. **Coverage (§4) is filled with real figures** measured
> from `data/djezzy_pages.json`. **Response quality (§2) and latency (§3) are left as
> `(from run)`** on purpose: they must come from a Colab T4 run of the *current* code
> (the off-domain guard and tier-ordering changed on 2026-06-10), and the on-disk
> `latency_store.json` holds only a stub record from a local mock — not real timings.
> To populate them, run `python test_scenarios.py` on Colab (it prints the per-scenario
> result + by-language/by-route summary and writes real `latency_store.json` records),
> then paste the figures into the marked cells. Nothing here is estimated — stale or
> guessed numbers are deliberately avoided in a measured-data deliverable.

---

## 1. Architecture overview

DjezzyBot is a retrieval-augmented chatbot: it answers strictly from text scraped
from Djezzy's own websites, never from the language model's own knowledge. (Djezzy
describes every offer in the page HTML, so plain text extraction is enough — no
image OCR is used.) It serves both text and voice, in Arabic, French, English, and
Algerian Darija.

### Modules

| Module | Responsibility |
|---|---|
| `config.py` | Single source of truth: model IDs, domains, every path and tunable. |
| `data/lexicon.py` | Offer names, synonyms, competitor/roaming/budget/comparison trigger words (FR/AR/Darija), Darija markers; `expand_synonyms`, `detect_offers`. |
| `scraper.py` | URL discovery (sitemap + Playwright BFS crawl + path hints) and HTML cleaning. Writes `data/djezzy_pages.json`. |
| `indexer.py` | Per-page chunking (1200/120), e5 embedding with `passage:`/`query:` prefixes + normalization, FAISS `IndexFlatIP`. |
| `retriever.py` | `smart_retrieve()` — the 7-route intent router in front of FAISS; presents priced offers in deterministic cheapest-first order (catalogue/budget/named/comparison), never relying on the LLM to sort. |
| `bot.py` | Qwen2.5-7B (4-bit NF4) loading, 11-rule system prompt (rule 1 = Djezzy-domain scope), language detection, manual history window, greedy generation, latency wrapper. |
| `voice.py` | Whisper-medium STT, Coqui XTTS-v2 TTS (on-demand), voice round-trip with per-stage latency. |
| `scheduler.py` | Daily 03:00 refresh (daemon thread) + `force_refresh()` for the UI button. |
| `app.py` | Gradio UI: single shared screen where text and voice feed one conversation, status bar, hot-swap on refresh. |
| `test_scenarios.py` | 14 acceptance scenarios; accumulates the latency this report uses. |

### Data flow (user input → response)

```
                 ┌──────────────────────────── text ───────────────────────────┐
 user types ─────┤                                                              │
                 │  bot.detect_language ─► retriever.smart_retrieve(q,lang,db)  │
 user speaks ─┐  │        │ COMPETITOR sentinel ─► canned refusal               │
              │  │        │ []  (no match)      ─► canned "not found"           │
 voice.transcribe (Whisper, auto-lang)          │ list[Document]                │
   │ text+lang─┘  │        ▼                                                     │
   │              │  bot.build_prompt(system rules + context + history[-8:] +   │
   │              │                   language directive, all in the USER turn) │
   │              │        ▼                                                     │
   │              │  Qwen2.5-7B greedy generate ─► answer (user's language)     │
   │              └──────────────────────────────────────────────────────────┬─┘
   │                                                                          │
   └─ voice: voice.synthesize(answer, lang) via XTTS-v2 ─► autoplay           ▼
                                                                    text shown in UI
 latency wrapper records retrieval / generation (text) and
 STT / retrieval / generation / TTS (voice) into bot.LATENCY → latency_store.json

 Daily 03:00 or "Refresh" button → scheduler.force_refresh →
   scraper.run_scrape → indexer.build_index → hot-swap the live index → log
```

### The 7 retrieval routes (priority order)

1. **COMPETITOR** — query names Ooredoo/Mobilis (Latin or Arabic) → sentinel → polite refusal.
2. **ROAMING** — destination/Hadj/Omra (incl. Arabic حج/عمرة) → k=6, roaming chunks preferred. Wins over budget so a foreign price is never filtered as national.
3. **CATALOGUE** — "your offers" with no specific offer → one exact-match chunk per known offer.
4. **COMPARISON** — comparison cue + ≥2 named offers → 4 chunks per offer, both guaranteed present.
5. **BUDGET** — amount + intent cue → pull a price pool, **filter prices in Python** (≤ budget), return ≤5. The LLM never filters.
6. **NAMED OFFER** — a single offer named → its whole page, all tiers, ordered
   cheapest-first in Python (the Cam Puce fix) + 2 dense chunks.
7. **NORMAL** — fallback dense search, k=5. Off-domain handling is two-layered: a
   low score floor (`OOD_MIN_SIMILARITY = 0.70`) only trips on degenerate input
   (gibberish, broken retrieval); topical off-domain questions (weather, jokes) are
   refused by the LLM's domain-scope rule (system-prompt rule 1), which judges by
   meaning. Calibration showed the embedding score can't separate off-domain from
   real questions (the bands overlap), so relevance is decided by the LLM, not a
   threshold — and a real customer is never refused on a borderline score.

(COMPETITOR and the empty/degenerate cases short-circuit to a canned refusal; the
other five routes feed retrieved context to the LLM, which also enforces rule 1.)

---

## 2. Response quality

Source: `python test_scenarios.py` (its printed summary).

**Overall accuracy:** _(from run)_  ___ / 14 passed.

### By scenario

| # | Scenario | Lang | Route | Pass? _(from run)_ |
|---|---|---|---|---|
| 01 | "vos offres ?" lists ≥4 offers with prices | fr | catalogue | |
| 02 | "j'ai 500 DA" shows only offers ≤500 DA | fr | budget | |
| 03 | "parle-moi de Campuce" — Campuce only, tiers cheapest-first | fr | named-offer | |
| 04 | "différence Legend / iZZY" — both with prices | fr | named-offer | |
| 05 | roaming France returns roaming, not national | fr | roaming | |
| 06 | English query → English reply | en | language | |
| 07 | Arabic query → Arabic reply | ar | language | |
| 08 | Darija query → detected, MSA Arabic reply | dz | language | |
| 09 | "offre Ooredoo ?" → polite Djezzy-only refusal | fr | competitor | |
| 10 | "la météo ?" → declined (LLM domain-scope rule), no weather | fr | out-of-domain | |
| 11 | Legend then "c'est combien ?" → still Legend | fr | context | |
| 12 | 6 turns, 7th still answers correctly | fr | context | |
| 13 | named offer returns its HTML price | fr | named-offer | |
| 14 | deep URL discovered automatically (crawler) | fr | coverage | |

### By language _(from run)_

| Language | Passed / total |
|---|---|
| Arabic (ar) | |
| French (fr) | |
| English (en) | |
| Darija (dz) | |

### By route _(from run)_

| Route | Passed / total |
|---|---|
| catalogue | |
| budget | |
| roaming | |
| named-offer | |
| competitor | |
| out-of-domain | |
| context | |
| coverage | |

---

## 3. Latency

Source: `latency_store.json` (written live by the `timed()` wrapper in `bot.py`,
one record per request, stages measured separately). Report average and worst-case
over the test run.

### Text response time _(from run)_

| Stage | Average (s) | Worst case (s) |
|---|---|---|
| Retrieval | | |
| Generation | | |
| **Total** | | |

### Voice round-trip _(from run)_

| Stage | Average (s) | Worst case (s) |
|---|---|---|
| STT (Whisper) | | |
| Retrieval | | |
| Generation | | |
| TTS (XTTS-v2) | | |
| **Total** | | |

> Note: the first voice request includes the one-time on-demand XTTS-v2 load, so
> its worst-case TTS figure is higher than the steady-state average. This is left
> in (not discarded) so the worst case is honest.

---

## 4. Coverage

Source: the crawl run — `scraper.run_scrape()` logs and `data/djezzy_pages.json`;
chunk count from `store.index.ntotal`.

| Metric | Value |
|---|---|
| Domains crawled | www.djezzy.dz, www.djezzy5g.dz |
| Pages kept (≥300 chars after cleaning) | **427** (all unique URLs) |
| — of which Arabic (`/ar/`) pages | **190** |
| — of which carrying a DA price | **117** |
| Total chunks indexed | ≈ **769** (estimate from page text; exact = `store.index.ntotal` after build) |
| Crawl stop reason | 45-min safety fuse (more URLs still queued — the site has many `/ar/` duplicates) |
| Offer categories represented | Legend ×10, Legend Pro ×3, Legend Max ×4, Cam Puce ×7, iZZY ×1, Zid ×5, Confort ×13, 3ayla ×2, Hayla ×1, Flexy ×25, 5G ×36, Roaming ×38 (pages mentioning each) |

---

## 5. Known limitations

An honest list of where the system is weak:

- **Offers rendered only as images.** Extraction is HTML-text only. Djezzy currently
  describes every offer in the page text (verified across the catalogue), but if a
  future promo published a price *exclusively* inside a graphic, it would be missed
  until the page's text is updated.
- **Darija coverage is lexical.** Language detection and synonym expansion rely on
  a hand-built Darija word list; unusual spellings or code-switching that isn't in
  the list may be classified as French. The reply is still grounded in context,
  but may come back in French instead of MSA Arabic.
- **No reranker / pure dense retrieval.** Per the fixed stack, retrieval is FAISS
  `IndexFlatIP` only. Routing precision therefore rests on the lexicon and the
  exact-match chunk selection; an offer named with a spelling not in `OFFER_NAMES`
  can fall through to the NORMAL route.
- **Off-domain refusal is LLM-judged, not a hard gate.** Calibration (≈95 queries,
  all 4 languages) showed the embedding similarity score cannot separate off-topic
  questions from real ones — e5 scores everything 0.77–0.85, and gibberish even
  scores *higher* than some real questions. So the score floor is set low (0.70) to
  catch only degenerate input, and topical off-domain (weather, jokes) is declined
  by the LLM's domain-scope rule. This is a probabilistic guard: a cleverly phrased
  off-topic question could occasionally get a partial answer. The trade is deliberate
  — it removes the false-refusal risk that a strict score gate placed on real
  customers (especially short Darija questions, which scored as low as 0.774).
- **Budget parsing scope.** Budgets are recognized via a dinar token (`DA`,
  `dinars`, `دج`). A bare number with no currency word, or budgets phrased only in
  words ("cinq cents dinars"), are not parsed as budget intent.
- **Crawl is a snapshot.** Data is only as fresh as the last successful daily
  refresh. If a refresh fails (site change, rate-limit, render timeout), the bot
  keeps serving the previous index — correct but potentially stale until the next
  run.
- **Roaming country breadth.** Roaming routing covers the destinations encoded in
  `ROAMING_TRIGGERS`. A country not in that map falls back to generic retrieval and
  may not surface the dedicated roaming tariff.
- **Single-GPU VRAM headroom.** Qwen-7B + e5 + Whisper are resident (~8 GB); XTTS
  loads on demand (~+2 GB). This fits a 15 GB T4 comfortably, but loading an
  additional large model alongside would risk OOM.
