"""
scheduler.py — daily knowledge refresh + manual force-refresh.

Keeps the bot's data current by re-running the scraper and rebuilding the FAISS
index once a day at SCHEDULE_TIME (03:00). On Colab the loop runs in a daemon
thread so it never blocks the Gradio app. The same refresh is exposed as
force_refresh() for the "Refresh" button in the UI.

Public API
----------
    force_refresh() -> dict          # {ok, n_pages, ts, message}
    start_scheduler(on_done=None)    # launch the daily 03:00 daemon thread

Both paths share _do_refresh(), which scrapes -> rebuilds -> hot-swaps the shared
index and appends a line to REFRESH_LOG. The newly built FAISS store is published
on the module attribute `CURRENT_INDEX`, and an optional callback lets app.py swap
in the new store without restarting.
"""

import time
import logging
import threading
from datetime import datetime, timezone

import config
import scraper
import indexer

logger = logging.getLogger("djezzybot.scheduler")

# The most recently built index; app.py reads this after a refresh.
CURRENT_INDEX = None
_lock = threading.Lock()


def _log_line(message: str):
    """Append a timestamped line to REFRESH_LOG (best-effort)."""
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with open(config.REFRESH_LOG, "a", encoding="utf-8") as f:
            f.write(f"{ts}\t{message}\n")
    except OSError as e:
        logger.warning("could not write refresh log: %s", e)


def _do_refresh() -> dict:
    """Scrape -> rebuild index -> hot-swap CURRENT_INDEX. Returns a status dict.

    Guarded by a lock so the daily job and a button press can't run concurrently.
    If the scrape yields no pages, the old index is kept (refresh treated as a
    no-op failure) so the bot never goes dark on a bad crawl.
    """
    global CURRENT_INDEX
    with _lock:
        ts = datetime.now(timezone.utc).isoformat()
        try:
            pages = scraper.run_scrape()
            if not pages:
                msg = "refresh FAILED: scraper returned 0 pages; keeping old index"
                logger.error(msg)
                _log_line(msg)
                return {"ok": False, "n_pages": 0, "ts": ts, "message": msg}

            store = indexer.build_index(pages)
            CURRENT_INDEX = store
            msg = f"refresh OK: {len(pages)} pages reindexed"
            logger.info(msg)
            _log_line(msg)
            return {"ok": True, "n_pages": len(pages), "ts": ts, "message": msg}
        except Exception as e:
            msg = f"refresh ERROR: {e}"
            logger.exception(msg)
            _log_line(msg)
            return {"ok": False, "n_pages": 0, "ts": ts, "message": msg}


def force_refresh() -> dict:
    """Run a refresh immediately (wired to the Gradio 'Refresh' button)."""
    logger.info("force_refresh requested")
    return _do_refresh()


def _scheduler_loop(on_done=None):
    """Background loop: run _do_refresh() every day at SCHEDULE_TIME."""
    import schedule

    def _job():
        result = _do_refresh()
        if on_done:
            try:
                on_done(result)
            except Exception:
                logger.exception("on_done callback failed")

    schedule.every().day.at(config.SCHEDULE_TIME).do(_job)
    logger.info("scheduler armed for daily %s", config.SCHEDULE_TIME)
    while True:
        schedule.run_pending()
        time.sleep(30)


def start_scheduler(on_done=None):
    """Start the daily-refresh daemon thread (returns the Thread object).

    `on_done(result_dict)` is invoked after each scheduled refresh so the caller
    (app.py) can publish the new index / update the status bar. The thread is a
    daemon so it dies with the Colab kernel and never blocks shutdown.
    """
    t = threading.Thread(target=_scheduler_loop, args=(on_done,),
                         daemon=True, name="djezzy-refresh")
    t.start()
    return t


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(force_refresh())
