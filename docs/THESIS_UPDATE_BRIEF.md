# Thesis Update Brief — Ground Truth for DjezzyBot

**Purpose.** This document is the *source of truth* for updating the existing master's
thesis (`Mémoire Master/main.tex`) to match the **real, functional DjezzyBot**. Paste it
into the new conversation at the start. Where the old thesis and this brief disagree,
**this brief (and the code) wins** — the old thesis was written before the system was
finished and describes modules that were never built.

> Companion files to read alongside this brief (they are the real implementation):
> `REPORT.md`, `config.py`, `retriever.py`, `bot.py`, `indexer.py`, `scraper.py`, `app.py`.

---

## 0. The task in one paragraph

We already wrote this thesis (in **French**) and the supervisor reviewed it. We are now
(a) **translating it to English** (the defense will be in English — this is approved),
(b) **keeping the same chapter/section structure** unless you have a genuinely better
proposal for a section's *content* (structure stays the same), and (c) **correcting every
claim so it matches the system we actually built.** The coding is not 100% finished, so
expect minor future edits — write so numbers and claims are easy to update later.

---

## 1. Hard writing rules (follow all)

1. **English, simple vocabulary.** We must understand and *defend* every sentence in an
   oral exam. Define each technical term in plain words the first time it appears. Prefer
   a short clear sentence over an impressive one.
2. **Human, not AI-sounding.** A previous draft scored ~45% on an AI detector; target is
   *below ~30%*, but do not torture the text chasing a number. The real levers:
   - Pack the text with **our specific data** (real numbers, real decisions, real failures).
     Generic prose is what detectors flag; concrete specifics read as human.
   - **Vary sentence and paragraph length.** Don't make every list a neat triplet.
   - **Ban these tells:** *moreover, furthermore, delve, leverage, robust, seamless,
     comprehensive, it is worth noting, in conclusion, in today's world, plays a crucial role.*
   - Narrate decisions in **first person plural**: "We first tried X; it ran out of GPU
     memory, so we switched to Y." Engineering stories are the hardest thing to fake.
   - The student will hand-edit a few paragraphs per chapter afterward; leave it natural.
3. **Real citations only.** LLMs invent references — **never** cite a paper you have not
   verified exists. Use the verified list in §6. One consistent style (**IEEE**). Every
   claim about a method (RAG, embeddings, quantization…) gets a real citation.
4. **Work section by section, with review.** Do **not** regenerate the whole `main.tex` —
   it will scramble the LaTeX and break compilation. Edit one section, keep it compiling
   under **XeLaTeX**, show it, move on. Do not touch the preamble/fonts/bibliography
   blindly; switching French→English means updating `polyglossia`/`babel` language and
   removing French typographic bits, nothing more.
5. **Make numbers easy to update.** Put volatile figures in LaTeX macros near the top,
   e.g. `\newcommand{\numPages}{427}`, `\newcommand{\numTests}{13/14}`, and reference the
   macro in the text. When the coding finishes, updates are one line.
6. **Keep product names in Latin script**: DjezzyBot, Legend, iZZY, Campuce, Zid, Confort.

---

## 2. The real system (plain facts — use these, not the old thesis)

DjezzyBot is a **multilingual voice + text RAG chatbot** for Djezzy (Algerian telecom). It
answers customer questions about offers/prices **only** from text scraped from Djezzy's own
websites — never from the model's own knowledge. Languages: **Arabic, French, English,
Algerian Darija** (Darija questions are answered in Modern Standard Arabic). It runs entirely
on **free, open-source models on a single Google Colab T4 GPU (15 GB)** — no paid APIs.

**Pipeline (end to end):**
1. **Input** — typed text, or speech transcribed by **Whisper-medium** (faster-whisper),
   with language auto-detected.
2. **Intent routing** — a rule-based router (`smart_retrieve`) classifies the question into
   one of **7 routes** *before* touching the vector store (see §3).
3. **Retrieval** — **multilingual-e5-base** embeds the query (`query:` prefix, L2-normalized);
   search is **FAISS `IndexFlatIP`** = cosine similarity. Per-page chunking (1200/120).
4. **Generation** — **Qwen2.5-7B-Instruct, 4-bit NF4 quantized**, greedy decoding, with an
   11-rule system prompt; retrieved context + recent history go in the *user* turn.
5. **Output** — text in the UI; for voice, **Coqui XTTS-v2** (loaded on demand) speaks it.
6. **Daily refresh** — a scheduler re-scrapes and rebuilds the index at 03:00; a button
   does it on demand.

**Fixed model stack (state it as a table in the thesis):**

| Layer | Choice | Why (use in the text) |
|---|---|---|
| LLM | Qwen2.5-7B-Instruct, 4-bit NF4 | Strong multilingual incl. Arabic; 4-bit fits the T4; greedy = reproducible |
| Embeddings | intfloat/multilingual-e5-base | One model covers FR/AR/EN/Darija; no translation needed |
| Vector DB | FAISS IndexFlatIP | Exact cosine search; small corpus (~769 chunks) needs no approximate index |
| STT | Whisper-medium | Robust multilingual speech recognition |
| TTS | Coqui XTTS-v2 (on demand) | Multilingual; loaded only during a voice reply to save VRAM |
| Scraping | Playwright + BeautifulSoup | Renders Djezzy's JS pages; cleans HTML to text |
| UI / orchestration | Gradio + LangChain | Single screen, text+voice share one conversation |

**VRAM budget (T4, 15 GB):** Qwen-7B 4-bit + e5 + Whisper resident ≈ 8 GB; XTTS on demand
≈ +2 GB; peak ≈ 10 GB — comfortable headroom.

---

## 3. The 7 retrieval routes (real behavior)

Priority order: **Competitor → Roaming → Catalogue → Comparison → Budget → Named → Normal.**

1. **Competitor** — names Ooredoo/Mobilis (Latin or Arabic) → polite "Djezzy only" refusal.
2. **Roaming** — destination / Hajj / Umrah (incl. Arabic حج/عمرة) → roaming chunks preferred;
   wins over budget so a foreign tariff is never filtered as a national price.
3. **Catalogue** — "your offers?" with no specific offer → one priced snippet per gamme,
   **sorted cheapest-first in Python.**
4. **Comparison** — comparison cue + ≥2 named offers → each offer's page, offers ordered
   cheapest-first.
5. **Budget** — amount + intent cue → pull price chunks, **filter prices in Python (≤ budget)**,
   return cheapest-first. The LLM never does the filtering. *(This is the weakest route —
   Djezzy pages mix subscription prices with per-unit tariffs; documented as a limitation.)*
6. **Named offer** — a single offer named → its whole page, **all tiers ordered cheapest-first
   in Python** (so e.g. Cam Puce's 7 paliers are always sorted), + a couple of dense chunks.
7. **Normal** — fallback dense search, k=5.

**Deterministic ordering principle (worth a paragraph):** any route that presents priced
offers orders them cheapest-first **in Python**, never relying on the LLM to sort. One shared
helper does the tier splitting and sorting for all routes.

**Off-domain handling — TWO layers (this is a real experiment, use it):**
- A low similarity floor (`OOD_MIN_SIMILARITY = 0.70`) only blocks *degenerate* input.
- Topical off-domain (weather, jokes) is refused by the **LLM's domain-scope rule**
  (system-prompt rule 1), which judges by *meaning*.
- **Why:** we calibrated the embedding score on ~95 queries in 4 languages and found the
  similarity score **cannot** separate off-topic from real questions — the bands overlap
  (real customers 0.774–0.846, off-topic 0.748–0.833), and *gibberish scores higher than
  some real questions*. So a score threshold is the wrong tool; the LLM decides by meaning.
  This finding is a genuine, defensible contribution — present the numbers in a table/figure.

---

## 4. Old-thesis claims that are WRONG — fix each

Cross-check **every** section against `REPORT.md`. These are the known mismatches:

| Old thesis says | Reality — rewrite to | 
|---|---|
| **"Moteur de Recherche Hybride"** (hybrid search engine) | There is **no hybrid retrieval, no BM25, no reranker.** It is **pure dense FAISS `IndexFlatIP`** in front of a **rule-based intent router** (the 7 routes). If "hybrid" meant "router + dense," rename it to avoid implying BM25/reranking. |
| **"Traducteur Darija"** (a Darija translation module) | There is **no translation layer.** Darija is handled by (1) multilingual-e5 embedding the Darija text directly, (2) a lexicon of Darija markers for language detection, (3) a prompt directive telling Qwen to answer in MSA. No intermediate translation happens. |
| Image **OCR / Tesseract** for offers inside images | **OCR was removed entirely.** Djezzy describes every offer in page **HTML text** (verified across the catalogue), so plain text extraction is enough. Remove all OCR/Tesseract content. |
| Smaller/earlier LLM, or an API model as the final choice | Final model is **local Qwen2.5-7B-Instruct, 4-bit NF4.** Chapter 5 may compare *API vs local* as a study — keep that comparison, but the **deployed system is local**. |
| Keyword-based out-of-domain refusal | Replaced by the **two-layer (score floor + LLM scope rule)** design in §3, after the calibration experiment. This is new — add it. |
| Any fixed "N pages/minute" crawl target | The crawl stops on **frontier exhaustion**; page/time caps are only **safety fuses**. |

If a section makes a claim you can't confirm from `REPORT.md`/code, **flag it for the student
to verify** rather than inventing a justification.

---

## 5. Real measured numbers (use these; mark with macros)

**Coverage (from `data/djezzy_pages.json`):**
- Pages kept (≥300 chars): **427** · Arabic `/ar/` pages: **190** · pages with a DA price: **117**
- Chunks indexed: **≈769** (exact = `store.index.ntotal` after build)
- Per-gamme page coverage: Legend ×10, Legend Pro ×3, Legend Max ×4, Cam Puce ×7, iZZY ×1,
  Zid ×5, Confort ×13, 3ayla ×2, Hayla ×1, Flexy ×25, 5G ×36, Roaming ×38
- Domains: www.djezzy.dz, www.djezzy5g.dz

**Acceptance tests (latest Colab run, `test_scenarios.py`): 13/14 (92%).**
- By language: FR 10/11, EN 1/1, AR 1/1, Darija 1/1
- By route: catalogue ✓, comparison ✓, roaming ✓, named-offer 3/3 ✓, competitor ✓,
  out-of-domain ✓, context 2/2 ✓, coverage ✓ — **budget 0/1** (the one fail; data-quality
  weakness, see §3 route 5).
- Deep URLs auto-discovered by the crawler: 131.

**Latency:** pull real per-stage numbers from `latency_store.json` produced by the run
(retrieval/generation for text; STT/retrieval/generation/TTS for voice). Note that the first
voice reply includes the one-time XTTS load, so its worst-case is higher than steady state.

---

## 6. Verified citations (real — IEEE style). Confirm each before use.

- Vaswani et al., "Attention Is All You Need," NeurIPS 2017. arXiv:1706.03762
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers," NAACL 2019. arXiv:1810.04805
- Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," NeurIPS 2020. arXiv:2005.11401
- Gao et al., "Retrieval-Augmented Generation for Large Language Models: A Survey," 2023. arXiv:2312.10997
- Karpukhin et al., "Dense Passage Retrieval for Open-Domain QA," EMNLP 2020. arXiv:2004.04906
- Reimers & Gurevych, "Sentence-BERT," EMNLP 2019. arXiv:1908.10084
- Wang et al., "Text Embeddings by Weakly-Supervised Contrastive Pre-training" (E5), 2022. arXiv:2212.03533
- Wang et al., "Multilingual E5 Text Embeddings: A Technical Report," 2024. arXiv:2402.05672
- Johnson, Douze, Jégou, "Billion-scale similarity search with GPUs" (FAISS), IEEE TBD 2019. arXiv:1702.08734
- Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," FnTIR 2009. (no arXiv — journal)
- Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale," 2022. arXiv:2208.07339
- Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs," 2023. arXiv:2305.14314
- Qwen Team, "Qwen2.5 Technical Report," 2024. arXiv:2412.15115
- Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper), 2022. arXiv:2212.04356
- Kim, Kong, Son, "Conditional VAE with Adversarial Learning for End-to-End TTS" (VITS), ICML 2021. arXiv:2106.06103
- Casanova et al., "XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model," 2024. arXiv:2406.04904
- Abid et al., "Gradio: Hassle-Free Sharing and Testing of ML Models," 2019. arXiv:1906.02569
- Abdul-Mageed et al., "ARBERT & MARBERT: Deep Bidirectional Transformers for Arabic," ACL 2021. arXiv:2101.01785 *(for Arabic/dialect NLP context)*

(LangChain, Playwright, bitsandbytes, Hugging Face Transformers, unsloth: cite as software /
documentation, not papers.)

---

## 7. Figures & tables to insert (with placeholders)

Leave a clearly-captioned placeholder for each; the student drops in the image later.

**Figures**
- System architecture / end-to-end data flow (text + voice).
- The 7-route intent-routing decision flow.
- RAG pipeline (scrape → chunk → embed → FAISS → retrieve → generate).
- **UI screenshots** — the single-screen Gradio app (text box + mic + spoken reply).
- Example conversations, one per language (FR / AR / EN / Darija) — real screenshots.
- Off-domain calibration: the overlapping score bands (real-customer vs off-topic vs gibberish).

**Tables**
- Model stack (component → choice → reason) — from §2.
- VRAM budget on the T4 — from §2.
- Acceptance-test results, by route and by language — from §5.
- Latency (text + voice round-trip) — from `latency_store.json`.
- Crawl coverage (pages, chunks, per-gamme) — from §5.
- Off-domain calibration numbers (in/off-domain score ranges) — from §3.

> Suggested new/expanded subsections in the **Results** chapter: a "Demonstration" section
> for the UI screenshots and example conversations, and an "Off-domain robustness" subsection
> presenting the calibration experiment.

---

## 8. Known limitations (be honest — examiners respect this)

- **Budget route is the weakest** — Djezzy pages mix subscription prices with per-unit
  tariffs and fees, so budget answers can be noisy (this is the one failing acceptance test).
- **Off-domain refusal is LLM-judged, not a hard gate** — a cleverly phrased off-topic
  question could occasionally get a partial answer (deliberate trade to avoid wrongly
  refusing real customers; the embedding score genuinely cannot separate the two).
- **Pure dense retrieval, no reranker** — routing precision rests on the lexicon + intent
  router; an offer named with an unknown spelling may fall through to the Normal route.
- **Darija coverage is lexical** — unusual spellings/code-switching not in the word list may
  be classified as French.
- **Snapshot data** — answers are only as fresh as the last successful daily crawl.
- **No OCR** — an offer published *only* inside an image (none today) would be missed.
