"""
scraper.py — Self-discovering Djezzy crawler with image OCR.

This is the data source for the whole bot. It must find offer pages BY ITSELF
(no manual URL list) and extract text that lives only inside images. Three
discovery layers are combined and deduplicated:

  (a) sitemaps      — parse /sitemap.xml and /sitemap_index.xml on every domain
  (b) BFS crawl     — render each page with Playwright (depth <= CRAWL_MAX_DEPTH),
                      collect every internal link. This is how deep pages such as
                      /entreprises/offres/djezzy-legend-pro/ are found even though
                      listing pages don't link to them cleanly.
  (c) path hints    — probe known path patterns (PATH_HINTS) on each domain.

For every kept page we also OCR its non-decorative images (Tesseract ara+fra+eng),
because many offers/prices appear only as graphics. OCR text is kept only when it
carries a signal (a digit, DA/Go/Mo, or a known offer name) and is appended under
an [IMAGE]: tag.

Public API
----------
    discover_urls() -> list[str]
    clean_html(html) -> str
    scrape_page(page, url) -> dict | None      # page = Playwright Page
    run_scrape() -> list[dict]                 # full pipeline, writes DATA_JSON

Each kept page is {title, url, content, has_ocr, scraped_at}.

Heavy deps (playwright, bs4, pytesseract, PIL, requests) are imported lazily inside
functions so that importing this module — and the pure helpers — stays cheap and
testable without a browser installed.
"""

import io
import os
import json
import time
import random
import logging
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse, urldefrag

import config
from data.lexicon import OFFER_NAMES

logger = logging.getLogger("djezzybot.scraper")

# Non-HTML asset extensions we never want to enqueue as crawlable pages.
_SKIP_EXT = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".css", ".js", ".zip", ".rar", ".mp4", ".mp3", ".doc", ".docx",
    ".xls", ".xlsx", ".ppt", ".pptx", ".woff", ".woff2", ".ttf",
)

# Host portion of each configured domain, used to keep the crawl on-site.
_ALLOWED_HOSTS = {urlparse(d).netloc.lower() for d in config.DOMAINS}

# OCR signal: any offer name (lowercased) also counts as a "keep" signal.
_OFFER_SIGNALS = [n.lower() for n in OFFER_NAMES]


# ===========================================================================
# Async-loop safety (Colab / Jupyter)
# ===========================================================================
def _loop_is_running() -> bool:
    """True if we're inside a running asyncio loop (Colab notebooks always are)."""
    import asyncio
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _maybe_thread(fn, *args, **kwargs):
    """Run `fn` in a worker thread when an asyncio loop is already running.

    Playwright's sync API raises if called inside a running event loop (the case
    in Colab/Jupyter). A fresh worker thread has no running loop, so the sync API
    works there. Outside a loop (plain `python scraper.py`) we just call directly.
    """
    if _loop_is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(fn, *args, **kwargs).result()
    return fn(*args, **kwargs)


# ===========================================================================
# URL helpers (pure)
# ===========================================================================
def _normalize_url(url: str) -> str:
    """Drop the fragment and trailing slash so we don't crawl the same page twice."""
    url, _ = urldefrag(url)
    if url.endswith("/") and len(urlparse(url).path) > 1:
        url = url[:-1]
    return url


def _is_internal(url: str) -> bool:
    """True if `url` is an http(s) page on one of the configured Djezzy hosts."""
    try:
        p = urlparse(url)
    except ValueError:
        return False
    if p.scheme not in ("http", "https"):
        return False
    if p.netloc.lower() not in _ALLOWED_HOSTS:
        return False
    if p.path.lower().endswith(_SKIP_EXT):
        return False
    return True


def _polite_sleep():
    """Random 1–2 s pause between requests to stay courteous to the site."""
    time.sleep(random.uniform(*config.CRAWL_DELAY))


# ===========================================================================
# Layer (a): sitemap discovery
# ===========================================================================
def parse_sitemaps() -> set:
    """Fetch each domain's sitemap(s) and return all internal page URLs found.

    Handles sitemap-index files (which point at child sitemaps) one level deep.
    Uses plain HTTP (requests) — sitemaps are static XML, no browser needed.
    """
    import requests
    from bs4 import BeautifulSoup

    found = set()
    headers = {"User-Agent": config.USER_AGENT}
    to_fetch = [d.rstrip("/") + path for d in config.DOMAINS for path in config.SITEMAP_PATHS]
    seen_sitemaps = set()

    while to_fetch:
        sm_url = to_fetch.pop()
        if sm_url in seen_sitemaps:
            continue
        seen_sitemaps.add(sm_url)
        try:
            resp = requests.get(sm_url, headers=headers, timeout=15)
            if resp.status_code != 200 or "xml" not in resp.headers.get("Content-Type", "") \
                    and "<urlset" not in resp.text and "<sitemapindex" not in resp.text:
                continue
            soup = BeautifulSoup(resp.text, "xml")
            # child sitemaps (sitemap index)
            for sm in soup.find_all("sitemap"):
                loc = sm.find("loc")
                if loc and loc.text.strip() not in seen_sitemaps:
                    to_fetch.append(loc.text.strip())
            # actual page URLs
            for url_tag in soup.find_all("url"):
                loc = url_tag.find("loc")
                if loc:
                    u = _normalize_url(loc.text.strip())
                    if _is_internal(u):
                        found.add(u)
        except Exception as e:
            logger.warning("sitemap fetch failed for %s: %s", sm_url, e)
        _polite_sleep()

    logger.info("sitemaps: %d URLs", len(found))
    return found


# ===========================================================================
# Layer (c): path-hint expansion
# ===========================================================================
def expand_path_hints() -> set:
    """Return domain × PATH_HINTS combinations as candidate seed URLs.

    These are not guaranteed to exist; they're seeds for the BFS crawl, which
    will silently drop any that fail to load. This guarantees the crawl starts
    from the offer/roaming sections even if the homepage doesn't link to them.
    """
    seeds = set()
    for d in config.DOMAINS:
        base = d.rstrip("/")
        seeds.add(_normalize_url(base))
        for hint in config.PATH_HINTS:
            seeds.add(_normalize_url(base + hint))
    return seeds


# ===========================================================================
# Layer (b): Playwright BFS crawl
# ===========================================================================
def _render(page, url: str) -> str | None:
    """Navigate Playwright `page` to `url` with retries; return HTML or None.

    Some Djezzy offer pages (e.g. iZZY) are JS SPAs that never fire `load` or
    reach `networkidle` (perpetual connections), so waiting on those returns an
    almost-empty shell. Instead we navigate to `domcontentloaded`, then wait
    until the page's visible text grows past RENDER_MIN_TEXT chars (i.e. JS has
    injected the real content), capped at RENDER_SETTLE_MS. Pages that are
    already full satisfy this instantly.
    """
    for attempt in range(1, config.CRAWL_RETRIES + 1):
        try:
            page.goto(url, timeout=config.PAGE_TIMEOUT_MS, wait_until="domcontentloaded")
            # wait for JS-injected content rather than network/load events
            try:
                page.wait_for_function(
                    "document.body && document.body.innerText.length > arguments[0]",
                    arg=config.RENDER_MIN_TEXT,
                    timeout=config.RENDER_SETTLE_MS,
                )
            except Exception:
                # page may legitimately be short, or JS stalled — take what we have
                pass
            page.wait_for_timeout(config.RENDER_EXTRA_SETTLE_MS)  # let images attach
            return page.content()
        except Exception as e:
            logger.debug("load attempt %d/%d failed for %s: %s",
                         attempt, config.CRAWL_RETRIES, url, e)
            time.sleep(config.CRAWL_RETRY_DELAY)
    logger.warning("giving up on %s after %d attempts", url, config.CRAWL_RETRIES)
    return None


def _links_from_html(html: str, base_url: str) -> set:
    """Extract normalized internal links from rendered HTML."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    out = set()
    for a in soup.find_all("a", href=True):
        u = _normalize_url(urljoin(base_url, a["href"]))
        if _is_internal(u):
            out.add(u)
    return out


def discover_urls() -> list:
    """Public entry point — runs the crawl in a thread when needed (Colab-safe)."""
    return _maybe_thread(_discover_urls_impl)


def _discover_urls_impl() -> list:
    """Combine all three discovery layers into a deduplicated, capped URL list.

    Runs the Playwright BFS crawl seeded by sitemap URLs + path hints, rendering
    each page to harvest new internal links up to CRAWL_MAX_DEPTH. Returns at most
    CRAWL_MAX_PAGES URLs. The rendered HTML is cached on each returned item's
    companion in `run_scrape`, but here we only return the URL list so the caller
    controls the page lifecycle.
    """
    from playwright.sync_api import sync_playwright

    seeds = parse_sitemaps() | expand_path_hints()
    logger.info("seeds before crawl: %d", len(seeds))

    visited = set()
    discovered = set(seeds)
    # BFS queue of (url, depth)
    queue = [(u, 0) for u in seeds]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=config.USER_AGENT)
        while queue and len(visited) < config.CRAWL_MAX_PAGES:
            url, depth = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            logger.info("[discover %d/%d] %s", len(visited),
                        config.CRAWL_MAX_PAGES, url)
            html = _render(page, url)
            _polite_sleep()
            if html is None or depth >= config.CRAWL_MAX_DEPTH:
                continue
            for link in _links_from_html(html, url):
                if link not in discovered:
                    discovered.add(link)
                    queue.append((link, depth + 1))
        browser.close()

    urls = sorted(discovered)[: config.CRAWL_MAX_PAGES]
    logger.info("discover_urls: %d URLs (visited %d during crawl)", len(urls), len(visited))
    return urls


# ===========================================================================
# HTML cleaning (pure)
# ===========================================================================
def clean_html(html: str) -> str:
    """Strip chrome/marketing noise and return readable text.

    Removes STRIP_TAGS entirely, plus any element whose class or id contains one
    of STRIP_CLASS_HINTS (menus, footers, cookie banners, modals, social...).
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(config.STRIP_TAGS):
        tag.decompose()

    # Remove elements whose class/id signals chrome/marketing noise. Two safety
    # rules learned the hard way:
    #   * Never decompose <html>/<body> — Bootstrap puts "modal-open" on <body>,
    #     and a naive match would wipe the entire page.
    #   * Match class/id *sub-tokens* (split on - and _), not raw substrings, so
    #     "modal" doesn't match "modal-open" loosely and "mega" doesn't eat
    #     "omega". A sub-token matches a hint if it starts with it (so "navbar",
    #     "navigation", "site-footer", "social-share" are all caught).
    # Decomposing a parent detaches its descendants; reaching one later makes
    # `.get` raise AttributeError (attrs cleared) — skip those, they're gone.
    hints = config.STRIP_CLASS_HINTS
    for el in soup.find_all(True):
        if el.name in ("html", "body"):
            continue
        try:
            tokens = list(el.get("class") or [])
            if el.get("id"):
                tokens.append(el.get("id"))
        except AttributeError:
            continue
        subtokens = [
            sub for tok in tokens
            for sub in tok.lower().replace("_", "-").split("-")
        ]
        if any(sub.startswith(h) for sub in subtokens for h in hints):
            el.decompose()

    text = soup.get_text(separator="\n")
    # collapse whitespace: trim lines, drop blanks, dedup consecutive blank runs
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def _page_title(html: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else ""


# ===========================================================================
# Image OCR
# ===========================================================================
def _is_decorative(src: str, width, height) -> bool:
    """True if an image looks decorative and should be skipped for OCR."""
    s = (src or "").lower()
    if any(tok in s for tok in config.OCR_SKIP_SRC):
        return True
    for dim in (width, height):
        try:
            if dim is not None and int(dim) < config.OCR_MIN_IMG_PX:
                return True
        except (TypeError, ValueError):
            pass
    return False


def _ocr_has_signal(text: str) -> bool:
    """Keep OCR text only if it carries a real signal (price/volume/offer name)."""
    t = text.lower()
    if any(ch.isdigit() for ch in t):
        return True
    if any(tok in t for tok in config.OCR_SIGNAL_TOKENS):   # da, go, mo
        return True
    if any(name in t for name in _OFFER_SIGNALS):
        return True
    return False


def _clean_ocr_text(text: str) -> str:
    """Tidy raw Tesseract output: collapse whitespace, drop 1–2 char noise lines."""
    lines = []
    for ln in text.splitlines():
        ln = " ".join(ln.split())
        if len(ln) >= 3:
            lines.append(ln)
    return "\n".join(lines)


_TESSERACT_OK = None  # cached availability of the tesseract binary


def _tesseract_available() -> bool:
    """True if the Tesseract binary is installed and callable (cached).

    OCR is optional: if Tesseract isn't present we skip image OCR entirely so the
    crawl doesn't waste time downloading images it can't read.
    """
    global _TESSERACT_OK
    if _TESSERACT_OK is None:
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            _TESSERACT_OK = True
        except Exception:
            _TESSERACT_OK = False
            logger.info("Tesseract not found — image OCR disabled (HTML text only)")
    return _TESSERACT_OK


def ocr_page_images(page_url: str, html: str) -> str:
    """OCR the non-decorative images on a page; return kept text (or "").

    For each candidate <img>: resolve its URL, download bytes, run Tesseract in
    ara+fra+eng, clean, and keep only if it has a signal. Kept blocks are joined.
    Returns "" immediately if Tesseract isn't installed (no image downloads).
    """
    if not _tesseract_available():
        return ""

    import requests
    import pytesseract
    from PIL import Image
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    headers = {"User-Agent": config.USER_AGENT}
    kept = []

    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or ""
        if not src:
            continue
        if _is_decorative(src, img.get("width"), img.get("height")):
            continue
        img_url = urljoin(page_url, src)
        try:
            resp = requests.get(img_url, headers=headers, timeout=15)
            if resp.status_code != 200:
                continue
            image = Image.open(io.BytesIO(resp.content))
            # second decorative check now that we know real pixel dimensions
            if _is_decorative(src, image.width, image.height):
                continue
            raw = pytesseract.image_to_string(image, lang=config.OCR_LANGS)
        except Exception as e:
            logger.debug("OCR failed for %s: %s", img_url, e)
            continue
        cleaned = _clean_ocr_text(raw)
        if cleaned and _ocr_has_signal(cleaned):
            kept.append(cleaned)

    return "\n".join(kept)


# ===========================================================================
# Per-page scrape
# ===========================================================================
def _scrape_from_html(url: str, html: str) -> dict | None:
    """Build a page dict from already-rendered HTML (clean + OCR), or None.

    Kept separate from rendering so the crawl can render a page once and both
    harvest its links and scrape its content from the same HTML.
    """
    content = clean_html(html)
    ocr_text = ocr_page_images(url, html)
    has_ocr = bool(ocr_text)
    if has_ocr:
        content = f"{content}\n{config.OCR_TAG} {ocr_text}".strip()

    if len(content) < config.MIN_PAGE_CHARS:
        return None

    return {
        "title": _page_title(html),
        "url": url,
        "content": content,
        "has_ocr": has_ocr,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def scrape_page(page, url: str) -> dict | None:
    """Render `url`, then clean + OCR it. Thin wrapper around _scrape_from_html.

    Returns None if the page fails to load or its content is shorter than
    MIN_PAGE_CHARS.
    """
    html = _render(page, url)
    if html is None:
        return None
    return _scrape_from_html(url, html)


# ===========================================================================
# Full pipeline
# ===========================================================================
def run_scrape() -> list:
    """Public entry point — runs the full scrape in a thread when needed (Colab-safe)."""
    return _maybe_thread(_run_scrape_impl)


def _run_scrape_impl() -> list:
    """Single-pass BFS crawl + scrape, rendering each page exactly once.

    For every page: render it, scrape its content from that HTML, and harvest its
    internal links to keep crawling — all from a single render (no second pass).
    Stops at CRAWL_MAX_PAGES rendered pages. Progress is logged per page so the
    run isn't silent. On a zero-page result the previous DATA_JSON cache is left
    untouched (callers treat an empty return as "refresh failed").
    """
    from playwright.sync_api import sync_playwright

    seeds = parse_sitemaps() | expand_path_hints()
    logger.info("seeds before crawl: %d", len(seeds))

    visited = set()
    discovered = set(seeds)
    queue = [(u, 0) for u in seeds]   # BFS queue of (url, depth)
    pages = []
    cap = config.CRAWL_MAX_PAGES

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=config.USER_AGENT)
        while queue and len(visited) < cap:
            url, depth = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)
            n = len(visited)

            html = _render(page, url)
            if html is None:
                logger.info("[%d/%d] skip (load failed)  %s", n, cap, url)
                _polite_sleep()
                continue

            # scrape from the HTML we already have — no second render
            try:
                rec = _scrape_from_html(url, html)
            except Exception as e:
                logger.warning("scrape error on %s: %s", url, e)
                rec = None
            if rec:
                pages.append(rec)
                logger.info("[%d/%d] kept  %s  (ocr=%s, %d chars)",
                            n, cap, url, rec["has_ocr"], len(rec["content"]))
            else:
                logger.info("[%d/%d] drop (too short)  %s", n, cap, url)

            # harvest links to continue the crawl (unless at max depth)
            if depth < config.CRAWL_MAX_DEPTH:
                new_links = 0
                for link in _links_from_html(html, url):
                    if link not in discovered:
                        discovered.add(link)
                        queue.append((link, depth + 1))
                        new_links += 1
                if new_links:
                    logger.info("        +%d new links (frontier=%d, discovered=%d)",
                                new_links, len(queue), len(discovered))
            _polite_sleep()
        browser.close()

    if pages:
        os.makedirs(os.path.dirname(config.DATA_JSON), exist_ok=True)
        with open(config.DATA_JSON, "w", encoding="utf-8") as f:
            json.dump(pages, f, ensure_ascii=False, indent=2)
        logger.info("run_scrape: rendered %d pages, kept %d (%d with OCR), "
                    "discovered %d URLs -> %s",
                    len(visited), len(pages),
                    sum(1 for pg in pages if pg["has_ocr"]),
                    len(discovered), config.DATA_JSON)
    else:
        logger.error("run_scrape: rendered %d pages but kept 0; "
                     "leaving previous cache untouched", len(visited))

    return pages


def load_pages() -> list:
    """Load the cached scraped pages from DATA_JSON (empty list if none yet)."""
    if not os.path.exists(config.DATA_JSON):
        return []
    with open(config.DATA_JSON, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    result = run_scrape()
    print(f"Scraped {len(result)} pages "
          f"({sum(1 for p in result if p['has_ocr'])} with OCR).")
