"""
diagnose_network.py — Phase 0 scraper reachability probe.

Run this from the SAME environment where the scraper fails (i.e. on Colab) to
find out WHY www.djezzy.dz is unreachable before changing any code.

It answers three questions:
  (a) Can plain `requests` (the library the sitemap/retry layer uses) reach the
      Djezzy site, with the configured User-Agent?
  (b) Can Playwright Chromium (a real browser) reach the same site?
  (c) Does this environment have working outbound HTTPS at all (example.com)?

Plus supporting signal: DNS resolution, redirect chain, non-www variant, and the
exact exception type when a fetch fails.

Usage on Colab:
    !python diagnose_network.py
(or paste the body into a cell). No GPU needed.
"""

import socket
import traceback

try:
    import config
    UA = config.USER_AGENT
    DJEZZY = "https://www.djezzy.dz"
except Exception:
    UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
    DJEZZY = "https://www.djezzy.dz"

LINE = "-" * 70


def hr(title):
    print(f"\n{LINE}\n{title}\n{LINE}")


# ---------------------------------------------------------------------------
# 0. DNS — can we even resolve the hostname?
# ---------------------------------------------------------------------------
def probe_dns(host):
    hr(f"0. DNS resolution for {host}")
    try:
        infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
        ips = sorted({i[4][0] for i in infos})
        print(f"  OK — resolves to: {ips}")
        return ips
    except Exception as e:
        print(f"  FAIL — {type(e).__name__}: {e}")
        return []


# ---------------------------------------------------------------------------
# Raw TCP connect to port 443 — isolates network reachability from TLS/HTTP
# ---------------------------------------------------------------------------
def probe_tcp(host, port=443, timeout=15):
    hr(f"0b. Raw TCP connect to {host}:{port}")
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        peer = s.getpeername()
        s.close()
        print(f"  OK — TCP handshake succeeded, peer={peer}")
        return True
    except Exception as e:
        print(f"  FAIL — {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# (a) + (c) requests
# ---------------------------------------------------------------------------
def probe_requests(url, label):
    hr(f"{label} requests.get — {url}")
    try:
        import requests
    except Exception as e:
        print(f"  requests not importable: {e}")
        return
    try:
        t0 = __import__("time").time()
        r = requests.get(url, headers={"User-Agent": UA}, timeout=30,
                         allow_redirects=True)
        dt = __import__("time").time() - t0
        print(f"  OK — status={r.status_code} in {dt:.1f}s")
        print(f"  final URL : {r.url}")
        if r.history:
            print(f"  redirects : {[h.status_code for h in r.history]} "
                  f"-> {[h.headers.get('Location') for h in r.history]}")
        print(f"  server    : {r.headers.get('Server')}")
        print(f"  bytes     : {len(r.content)}")
    except Exception as e:
        print(f"  FAIL — {type(e).__name__}: {e}")
        # show the underlying cause for ConnectTimeout / SSLError etc.
        cause = getattr(e, "__cause__", None) or getattr(e, "args", None)
        print(f"  detail    : {cause}")


# ---------------------------------------------------------------------------
# (b) Playwright
# ---------------------------------------------------------------------------
def probe_playwright(url, label):
    hr(f"{label} Playwright Chromium — {url}")
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:
        print(f"  playwright not importable: {e}")
        return

    # run in a worker thread so this works inside Colab's asyncio loop
    import concurrent.futures

    def _go():
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=UA)
            try:
                resp = page.goto(url, timeout=30000, wait_until="domcontentloaded")
                status = resp.status if resp else None
                title = page.title()
                html_len = len(page.content())
                browser.close()
                return ("OK", status, title, html_len, None)
            except Exception as ex:
                browser.close()
                return ("FAIL", None, None, None, f"{type(ex).__name__}: {ex}")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            result, status, title, html_len, err = ex.submit(_go).result()
        if result == "OK":
            print(f"  OK — HTTP status={status}")
            print(f"  title     : {title!r}")
            print(f"  html bytes: {html_len}")
        else:
            print(f"  FAIL — {err}")
    except Exception as e:
        print(f"  FAIL (launch) — {type(e).__name__}: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
def main():
    print("DjezzyBot Phase-0 network diagnostic")
    print(f"User-Agent: {UA}")

    # supporting signal
    probe_dns("www.djezzy.dz")
    probe_tcp("www.djezzy.dz")

    # (c) baseline: general outbound HTTPS
    probe_requests("https://example.com", "(c1)")
    probe_playwright("https://example.com", "(c2)")

    # (a) Djezzy via requests — www and non-www
    probe_requests(DJEZZY, "(a1)")
    probe_requests("https://djezzy.dz", "(a2 non-www)")

    # (b) Djezzy via Playwright
    probe_playwright(DJEZZY, "(b)")

    hr("DONE — paste this entire output back for diagnosis")


if __name__ == "__main__":
    main()
