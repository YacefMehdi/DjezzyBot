# DjezzyBot

A multilingual **voice + text RAG chatbot** for **Djezzy** (Algerian telecom). It
answers customer questions about offers, prices, roaming and services **strictly
from text scraped off Djezzy's own websites** — never from the language model's
own knowledge — in **Arabic, French, English and Algerian Darija** (Darija is
answered in Modern Standard Arabic).

Built to run on a single **Google Colab T4 (15 GB VRAM)**, open-source only, no
paid APIs.

## How it works

```
voice ─► Whisper (STT) ─┐
                        ├─► smart_retrieve (7-route intent router) ─► FAISS (e5)
text ──────────────────┘                        │
                                                 ▼
                              Qwen2.5-7B (4-bit) grounded answer
                                                 │
                        text ◄───────────────────┴──────────────► XTTS-v2 (TTS) ─► voice
```

### Stack

| Layer | Choice |
|---|---|
| LLM | `Qwen2.5-7B-Instruct` (4-bit NF4, greedy) |
| Embeddings | `intfloat/multilingual-e5-base` (`passage:`/`query:` prefixes, L2-normalized) |
| Vector store | FAISS `IndexFlatIP` (cosine) — no reranker |
| STT | faster-whisper `medium` (float16) |
| TTS | Coqui XTTS-v2 (loaded on demand) |
| Scraping | Playwright + BeautifulSoup + Tesseract OCR |
| UI / orchestration | Gradio + LangChain |

### The 7 retrieval routes

`COMPETITOR → ROAMING → CATALOGUE → COMPARISON → BUDGET → NAMED OFFER → NORMAL`
(NORMAL has an out-of-domain guard). Budget filtering is done **in Python**, never
delegated to the LLM; competitor and off-topic questions short-circuit to a canned
refusal.

## Files

| File | Responsibility |
|---|---|
| `config.py` | Single source of truth (model IDs, domains, paths, tunables) |
| `data/lexicon.py` | Offer names, synonyms, competitor/roaming/budget/comparison cues (FR/AR/Darija) |
| `scraper.py` | Self-discovering crawler (sitemap + BFS + path hints) + image OCR |
| `indexer.py` | Chunking + e5 embeddings + FAISS build/load |
| `retriever.py` | `smart_retrieve()` — the 7-route intent router |
| `bot.py` | LLM load, prompt assembly, language detection, history, latency |
| `voice.py` | Whisper STT + XTTS-v2 TTS round trip |
| `scheduler.py` | Daily 03:00 refresh + manual force-refresh |
| `app.py` | Gradio UI (text + voice tabs) |
| `test_scenarios.py` | 14 acceptance tests + latency capture |
| `DjezzyBot.ipynb` | Colab notebook (installs + run order) |
| `REPORT.md` | Architecture + measured performance report |

## Running on Colab

1. Upload this folder (the cached `data/djezzy_pages.json` is included).
2. Open `DjezzyBot.ipynb` and run the install cells.
3. Build the index **from the cached scrape** (the Djezzy site geo-blocks Colab's
   datacenter IP, so don't crawl from Colab):
   ```python
   import scraper, indexer
   indexer.build_index(scraper.load_pages())
   ```
4. Launch the app:
   ```python
   import app; app.main()
   ```
5. (Optional) Run the acceptance suite: `python test_scenarios.py`.

> Refreshing the data (scheduler / refresh button) must be run from an Algerian IP;
> on Colab the bot serves the cached snapshot.

## Status

Code complete and logic-verified. Performance figures in `REPORT.md` marked
*(from run)* are filled from a real Colab T4 run.
