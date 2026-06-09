"""
config.py — Single source of truth for DjezzyBot.

Every model ID, domain, path, and tunable parameter lives here. No other module
should hardcode any of these values; they import from `config` instead. This is
what lets the daily refresh, the Colab notebook, and the test suite all agree on
the same settings.
"""

import os

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
MAX_CONTEXT_CHARS = 6000          # cap on the retrieved context fed to the LLM

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
CRAWL_MAX_PAGES = 200            # hard cap after dedup
CRAWL_DELAY = (1.0, 2.0)         # polite random sleep (seconds) between requests
CRAWL_RETRIES = 3                # retry a failed page load this many times
CRAWL_RETRY_DELAY = 2.0          # seconds between retries
PAGE_TIMEOUT_MS = 30000          # Playwright per-page navigation timeout
# SPA rendering: wait until JS injects real content rather than load/networkidle
# events (which never fire on some Djezzy offer pages like iZZY).
RENDER_MIN_TEXT = 400            # body innerText length that signals "content present"
RENDER_SETTLE_MS = 10000         # max wait for that content to appear
RENDER_EXTRA_SETTLE_MS = 1500    # extra pause so lazy <img> tags attach (for OCR)

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
# Image OCR
# ---------------------------------------------------------------------------
OCR_LANGS = "ara+fra+eng"        # Tesseract language packs
OCR_MIN_IMG_PX = 200             # skip images smaller than this (decorative)
# Skip images whose src contains any of these (logos, icons, sprites...).
OCR_SKIP_SRC = ["logo", "icon", "sprite", "avatar", "flag", "social"]
# Keep OCR text only if it contains one of these signals: any digit, or one of
# these tokens, or a known offer name (offer names come from data/lexicon.py).
OCR_SIGNAL_TOKENS = ["da", "go", "mo"]
OCR_TAG = "[IMAGE]:"             # OCR text is appended to page content under this tag

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
# Out-of-domain guard. Primary gate is the telecom-signal check on the query; this
# is a secondary backstop — a signal-bearing query whose best chunk scores below
# this cosine similarity is also treated as off-topic. Tune on Colab if needed.
OOD_MIN_SIMILARITY = 0.50

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
