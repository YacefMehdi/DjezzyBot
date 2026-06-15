"""
config.py — Single source of truth for DjezzyBot.

Every model ID, domain, path, and tunable parameter lives here. No other module
should hardcode any of these values; they import from `config` instead. This is
what lets the daily refresh, the Colab notebook, and the test suite all agree on
the same settings.
"""

import os

# ---------------------------------------------------------------------------
# CUDA memory — tight 15 GB T4 budget
# ---------------------------------------------------------------------------
# Qwen-7B (4-bit) + Whisper-medium + e5 + XTTS-v2 all share one T4. The pieces fit,
# but PyTorch's default caching allocator fragments the heap, so loading XTTS AFTER
# Qwen has generated could fail with "CUDA out of memory" on a heap that has enough
# FREE bytes but no single contiguous block. expandable_segments lets the allocator
# grow a segment instead of demanding one contiguous slab, which removes that failure.
# Must be set BEFORE torch initialises CUDA — config is imported before any torch use.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# BASE_DIR is the folder containing this file. Everything else is relative to it
# so the project works identically on Colab (/content/...) and locally.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

DATA_JSON = os.path.join(DATA_DIR, "djezzy_pages.json")   # scraped pages cache
FAISS_DIR = os.path.join(DATA_DIR, "faiss_index")          # persisted vector store
REFRESH_LOG = os.path.join(BASE_DIR, "refresh_log.txt")    # scheduler / refresh log
LATENCY_STORE = os.path.join(BASE_DIR, "latency_store.json")  # accumulated timings

# ---------------------------------------------------------------------------
# Model IDs  (fixed stack — do not substitute)
# ---------------------------------------------------------------------------
LLM_MODEL_ID = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"        # 4-bit NF4, greedy
EMBED_MODEL_ID = "intfloat/multilingual-e5-base"            # needs query:/passage:
STT_MODEL_ID = "openai/whisper-medium"                      # auto language detect
STT_COMPUTE_TYPE = "float16"   # faster-whisper compute type on GPU (~1.5GB vs fp32 ~3GB)
TTS_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"  # Coqui XTTS-v2

# ---------------------------------------------------------------------------
# Embedding / E5 conventions
# ---------------------------------------------------------------------------
# multilingual-e5 REQUIRES these prefixes and L2-normalized vectors.
E5_PASSAGE_PREFIX = "passage: "   # prepended to documents at indexing time
E5_QUERY_PREFIX = "query: "       # prepended to search strings at retrieval time
EMBED_NORMALIZE = True            # FAISS IndexFlatIP == cosine when normalized

# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------
MAX_NEW_TOKENS = 768
DO_SAMPLE = False                 # greedy / deterministic
REPETITION_PENALTY = 1.1
MAX_CONTEXT_CHARS = 8000          # cap on the retrieved context fed to the LLM
                                  # (fits a full named-offer page, or every gamme's
                                  #  snippet in a catalogue listing)

# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------
# Plain Python list, sliding window of the last 4 exchanges (history[-8:]),
# managed manually — NOT LangChain memory.
HISTORY_WINDOW = 8                # 4 user + 4 assistant messages

# ---------------------------------------------------------------------------
# Scraper — domains & crawl
# ---------------------------------------------------------------------------
DOMAINS = [
    "https://www.djezzy.dz",
    "https://www.djezzy5g.dz",
]

# Layer (a): sitemap entry points probed on each domain.
SITEMAP_PATHS = ["/sitemap.xml", "/sitemap_index.xml"]

# Layer (b): recursive BFS crawl with Playwright.
CRAWL_MAX_DEPTH = 4
# The crawl's REAL stop is frontier-exhaustion: it ends when there are no new
# internal links left to follow, which self-sizes to the site (add/remove pages and
# it adjusts with zero edits). The two limits below are safety FUSES, not targets —
# they exist only to escape a pathological infinite crawl and should never bind in
# normal operation, so you don't re-tune them as offers come and go.
CRAWL_MAX_PAGES = 1500           # fuse: absolute ceiling on pages rendered
CRAWL_MAX_MINUTES = 45           # fuse: wall-clock budget for the whole crawl
CRAWL_DELAY = (1.0, 2.0)         # polite random sleep (seconds) between requests
CRAWL_RETRIES = 3                # retry a failed page load this many times
CRAWL_RETRY_DELAY = 2.0          # seconds between retries
PAGE_TIMEOUT_MS = 30000          # Playwright per-page navigation timeout
# SPA rendering: wait until JS injects real content rather than load/networkidle
# events (which never fire on some Djezzy offer pages like iZZY).
RENDER_MIN_TEXT = 400            # body innerText length that signals "content present"
RENDER_SETTLE_MS = 10000         # max wait for that content to appear
RENDER_EXTRA_SETTLE_MS = 1500    # extra settle pause so late-rendering content finishes

# Layer (c): path-hint expansion. Deep offer pages aren't cleanly linked from
# listing pages, so we also probe these known patterns on each domain.
PATH_HINTS = [
    "/particuliers/offres/",
    "/particuliers/services/",
    "/entreprises/offres/",
    "/entreprises/services/",
    "/offres/",
    "/roaming/",
    "/5g/",
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# ---------------------------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------------------------
# Whole tags removed outright.
STRIP_TAGS = [
    "script", "style", "header", "footer", "nav",
    "aside", "noscript", "iframe", "form",
]
# Elements whose class or id contains any of these substrings are removed
# (chrome/navigation/marketing noise).
STRIP_CLASS_HINTS = [
    "menu", "nav", "footer", "sidebar", "breadcrumb", "cookie",
    "popup", "modal", "overlay", "social", "sticky", "banner", "mega",
]
MIN_PAGE_CHARS = 300             # drop pages shorter than this after cleaning

# ---------------------------------------------------------------------------
# Chunking  (RecursiveCharacterTextSplitter; never cross page boundaries)
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 120

# ---------------------------------------------------------------------------
# Retriever — per-mode k values
# ---------------------------------------------------------------------------
K_ROAMING = 6
K_CATALOGUE_PER_OFFER = 1        # one exact-match chunk per known offer
K_COMPARISON_PER_OFFER = 4       # chunks per named offer in a comparison
K_BUDGET_POOL = 20               # price chunks pulled before the Python price filter
K_BUDGET_RETURN = 5              # offers returned after filtering
K_NAMED_EXACT = 3                # exact-match chunks for a named offer
K_NAMED_FAISS = 2                # FAISS chunks added for a named offer
K_NORMAL = 5
COMPETITOR_SENTINEL = "COMPETITOR"
# Out-of-domain floor — a near-zero tripwire only. The LLM is the SOLE off-domain judge.
# CALIBRATED twice on the real index (2026-06-10, ~95 queries, all 4 languages). The wide
# run proved a score threshold CANNOT judge relevance with this embedding model: e5 hands
# out high cosine scores to EVERYTHING. The three groups overlap completely in 0.77–0.85 —
#   real signal-less customers : 0.774–0.846   (lowest = Darija "شحال تدير فالشهر")
#   off-topic noise (jokes...)  : 0.748–0.833
#   pure GIBBERISH (keyboard mash): 0.798–0.838  ← scores HIGHER than real questions!
# So relevance is decided by MEANING (bot.SYSTEM_PROMPT rule 1, the LLM), never by score.
# This floor is set far below everything observed so it NEVER fires on real traffic; it only
# trips if retrieval is so broken the best match scores < 0.70 (e.g. an embedding failure).
# Any value in [0, ~0.74] is behaviourally identical here — 0.70 just keeps a tiny tripwire.
OOD_MIN_SIMILARITY = 0.70

# Dedicated binary "is this about Djezzy?" gate (bot._in_domain), run on the NORMAL
# route before generating. DISABLED by default (user decision): for a customer-facing
# voice assistant, never wrongly refusing a real subscriber outweighs catching every
# off-topic question, and OOD detection that is both high-recall and false-refusal-free
# is an open problem with this multilingual embedder. OOD defence falls to system-prompt
# rule 1 + the similarity floor. The full gate (plus the signal-skip / arithmetic-guard
# machinery in bot.generate_answer) stays behind this flag so the thesis can report the
# experiment and its trade-off; flip to True to reproduce the gated numbers.
OOD_GATE_ENABLED = False

# ---------------------------------------------------------------------------
# Voice
# ---------------------------------------------------------------------------
# XTTS-v2 supported language codes. Darija has no XTTS voice → speak it as Arabic.
TTS_LANG_MAP = {"fr": "fr", "en": "en", "ar": "ar", "dz": "ar"}
# Whisper restricts auto-detection to these to avoid mis-classification.
STT_ALLOWED_LANGS = ["fr", "ar", "en"]

# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
SCHEDULE_TIME = "03:00"          # daily refresh time (Colab local clock)

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
DJEZZY_RED = "#E4002B"
DJEZZY_WHITE = "#FFFFFF"
GRADIO_SHARE = True
