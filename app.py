"""
app.py — Gradio front-end (text + voice) and application wiring.

Boots the bot: loads the FAISS index (building it from the cached scrape if
needed), starts the daily-refresh scheduler, and serves a SINGLE-SCREEN Gradio UI
in Djezzy red/white where text and voice share one conversation:

    one chat window (shared history) + a text box + a microphone + spoken reply,
    plus clear / refresh and a status bar.

Typing answers in text; speaking transcribes the question into the same chat and
also plays the answer aloud. Language is auto-detected from the input (no manual
selector). Launches with share=True for Colab.

Run:  python app.py
"""

import logging
import os
import re
import traceback
from datetime import datetime, timedelta, timezone

import config
import scraper
import indexer
import scheduler
import bot
import voice

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("djezzybot.app")

# ---------------------------------------------------------------------------
# Shared application state
# ---------------------------------------------------------------------------
STATE = {
    "index": None,          # the live FAISS store
    "doc_count": 0,         # vectors indexed (for the status bar)
    "last_refresh": "never",
}


def _index_size(store) -> int:
    try:
        return store.index.ntotal
    except Exception:
        return 0


def boot():
    """Load (or build) the index and arm the scheduler. Called once at startup."""
    store = indexer.load_index()
    if store is None:
        logger.info("no FAISS index found — building from cached scrape")
        pages = scraper.load_pages()
        if not pages:
            logger.warning("no cached pages either — running a fresh scrape")
            pages = scraper.run_scrape()
        if pages:
            store = indexer.build_index(pages)
    STATE["index"] = store
    STATE["doc_count"] = _index_size(store) if store else 0
    STATE["last_refresh"] = "startup (cached)" if store else "never"
    scheduler.CURRENT_INDEX = store

    # when the daily job finishes, publish the fresh index into STATE
    def _on_refresh(result):
        if result.get("ok") and scheduler.CURRENT_INDEX is not None:
            STATE["index"] = scheduler.CURRENT_INDEX
            STATE["doc_count"] = _index_size(STATE["index"])
            STATE["last_refresh"] = result.get("ts", "")
    scheduler.start_scheduler(on_done=_on_refresh)
    logger.info("boot complete: %d docs indexed", STATE["doc_count"])


def _status_text() -> str:
    return (f"🟢 Base Djezzy — {STATE['doc_count']} documents indexés · "
            f"dernière mise à jour : {STATE['last_refresh']}")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
# --- message timestamps (texting-app style) --------------------------------
# Algeria is UTC+1 year-round (no DST); Colab runs in UTC, so we offset explicitly
# rather than trust the server clock. The stamp is appended to the DISPLAYED message
# as a small muted line and stripped back out before the text reaches the LLM, so it
# never pollutes the prompt.
_DZ_TZ = timezone(timedelta(hours=1))
_TS_RE = re.compile(r"\n\n<sub>.*?</sub>\s*$", re.DOTALL)
# Transient "bot is typing" bubble, yielded while the (slow) answer is generated so the
# screen isn't frozen after we removed Gradio's queue spinner. Never persisted/sent to LLM.
_TYPING = "*✍️ DjezzyBot rédige…*"


def _stamp() -> str:
    return datetime.now(_DZ_TZ).strftime("%H:%M · %d/%m")


def _with_ts(text: str) -> str:
    """Append a muted timestamp line to a displayed message."""
    return f"{text}\n\n<sub>{_stamp()}</sub>"


def _strip_ts(text):
    """Remove the appended timestamp so the LLM sees the clean message text."""
    return _TS_RE.sub("", text) if isinstance(text, str) else text


def _history_to_messages(chat_history):
    """Convert Gradio 'messages' history to the bot's role/content list.

    Audio bubbles (content is a {"path": ...} dict for the recorded question or the
    spoken answer) are SKIPPED, and the appended timestamp is stripped — only the
    clean text turns are sent to the LLM, so neither media nor timestamps pollute
    the prompt.
    """
    msgs = []
    for m in chat_history or []:
        if (isinstance(m, dict) and m.get("role") in ("user", "assistant")
                and isinstance(m.get("content"), str)):
            content = _strip_ts(m["content"]).strip()
            if content:
                msgs.append({"role": m["role"], "content": content})
    return msgs


def add_user_text(message, chat_history):
    """Phase 1 (instant): echo the typed message into the chat and clear the box.

    Split from the generation step so the user SEES their question and an empty
    textbox immediately, instead of waiting for the whole answer before anything
    appears. Returns (chat, cleared-textbox).
    """
    chat_history = chat_history or []
    if message and message.strip():
        chat_history = chat_history + [{"role": "user", "content": _with_ts(message)}]
    return chat_history, ""


def reply_text(chat_history, speak):
    """Phase 2 (slow): answer the last user message already shown in the chat.

    A GENERATOR: it first yields a transient "typing…" bubble (so the screen isn't
    frozen during the long generation now that the queue spinner is hidden), then
    yields the real answer. If `speak` is on, the reply is also synthesized and
    appended as a playable audio bubble so it STAYS in the conversation. Yields
    (chat, spoken-reply wav).
    """
    chat_history = chat_history or []
    if not chat_history or chat_history[-1].get("role") != "user":
        yield chat_history, None
        return
    message = _strip_ts(chat_history[-1].get("content"))
    if not isinstance(message, str) or not message.strip():
        yield chat_history, None
        return
    if STATE["index"] is None:
        yield chat_history + [{"role": "assistant",
                               "content": "⚠️ La base n'est pas encore indexée. "
                                          "Cliquez sur « Rafraîchir »."}], None
        return
    yield chat_history + [{"role": "assistant", "content": _TYPING}], None   # typing…
    prior = _history_to_messages(chat_history[:-1])     # everything before this question
    result = bot.answer(message, STATE["index"], prior)
    out = chat_history + [{"role": "assistant", "content": _with_ts(result["text"])}]
    wav = voice.synthesize(result["text"], result["lang"]) if speak else None
    if wav:
        out = out + [{"role": "assistant", "content": {"path": wav}}]
    yield out, wav


def on_clear():
    """Clear the conversation (and any pending spoken reply)."""
    return [], "", None


def on_refresh():
    """Refresh button: re-scrape + reindex, then update STATE + status bar."""
    result = scheduler.force_refresh()
    if result.get("ok"):
        STATE["index"] = scheduler.CURRENT_INDEX
        STATE["doc_count"] = _index_size(STATE["index"])
        STATE["last_refresh"] = result.get("ts", "")
    return _status_text()


def on_voice(audio_path, chat_history):
    """Transcribe speech → answer → speak, into the SAME shared conversation.

    Both audios STAY in the chat as playable bubbles: the recording you sent (so you
    can hear what you said) and the spoken answer (so you can replay any past reply,
    not just the latest). The transcription is shown as text under your recording.
    Returns (updated chat, spoken-reply wav path) — the wav also auto-plays once.
    """
    chat_history = chat_history or []
    if audio_path is None:
        yield chat_history, None
        return
    rec = {"role": "user", "content": {"path": audio_path}}          # your recording (replayable)
    if STATE["index"] is None:
        yield chat_history + [rec, {"role": "assistant",
                                    "content": "⚠️ La base n'est pas encore indexée. "
                                               "Cliquez sur « Rafraîchir »."}], None
        return
    # immediately show the recording + a typing bubble while we transcribe & answer
    yield chat_history + [rec, {"role": "assistant", "content": _TYPING}], None
    prior = _history_to_messages(chat_history)
    try:
        result = voice.voice_answer(audio_path, STATE["index"], prior)
    except Exception as e:
        # Gradio shows only a generic "erreur"; surface the REAL error (type + message
        # + the failing stage) into the chat AND the server log so it can be diagnosed
        # without hunting the Colab cell output.
        logging.getLogger("djezzybot.app").exception("voice path failed")
        tb = traceback.format_exc().strip().splitlines()
        where = tb[-1] if tb else f"{type(e).__name__}: {e}"
        yield chat_history + [rec, {"role": "assistant",
                                    "content": f"⚠️ Erreur vocale — {where}"}], None
        return
    msgs = chat_history + [
        rec,
        {"role": "user", "content": _with_ts(f"🎙️ {result['transcription']}")},
        {"role": "assistant", "content": _with_ts(result["text"])},
    ]
    # Only add the spoken-reply bubble if TTS actually produced audio; if VRAM was too
    # tight even for the CPU fallback, the text answer still stands (no error bubble).
    if result.get("wav_path"):
        msgs = msgs + [{"role": "assistant", "content": {"path": result["wav_path"]}}]
    yield msgs, result.get("wav_path")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
_CSS = f"""
.gradio-container {{ font-family: 'Segoe UI', sans-serif; }}
#title {{ color: {config.DJEZZY_RED}; font-weight: 800; }}
.djezzy-status {{ background: {config.DJEZZY_RED}; color: {config.DJEZZY_WHITE};
                  padding: 8px 14px; border-radius: 8px; font-weight: 600; }}
footer {{ visibility: hidden; }}

/* Bidirectional text. Each message follows its OWN script's direction: an Arabic or
   Darija reply renders right-to-left (so embedded Latin names like "Djezzy"/"iZZY" and
   short codes like "#121*" keep the correct visual order), while French/English stay
   left-to-right. `unicode-bidi: plaintext` applies the Unicode bidi algorithm per
   paragraph using its first strong character, so no per-message language flag is needed. */
.gradio-container .message,
.gradio-container .message *,
.gradio-container [class*="bubble"],
.gradio-container [class*="message"] p,
.gradio-container [class*="message"] li,
.gradio-container [class*="message"] span {{
  unicode-bidi: plaintext;
  text-align: start;
}}
"""


def build_ui():
    """Construct and return the Gradio Blocks app (not launched)."""
    import gradio as gr

    theme = gr.themes.Soft(primary_hue=gr.themes.colors.red,
                          neutral_hue=gr.themes.colors.gray)

    with gr.Blocks(theme=theme, css=_CSS, title="DjezzyBot") as demo:
        gr.Markdown("# 📱 DjezzyBot", elem_id="title")
        gr.Markdown("Assistant virtuel multilingue de Djezzy — écrivez **ou** parlez, "
                    "dans une seule conversation (Arabe · Français · English · Darija).")
        status = gr.Markdown(_status_text(), elem_classes=["djezzy-status"])

        # One shared conversation for BOTH text and voice → a single history.
        # Tall so the conversation fills the screen instead of a cramped scroller.
        chatbot = gr.Chatbot(type="messages", height=600, label="Conversation",
                             show_copy_button=True)

        with gr.Row():
            txt = gr.Textbox(
                placeholder="Écrivez votre question…  (ou parlez avec le micro ci-dessous)",
                scale=7, show_label=False, autofocus=True)
            send_btn = gr.Button("Envoyer", variant="primary", scale=1, min_width=110)
            stop_btn = gr.Button("⏹️ Stop", variant="stop", scale=1, min_width=90)

        with gr.Row():
            mic = gr.Audio(sources=["microphone"], type="filepath",
                           label="🎙️ Parler à DjezzyBot", scale=2)
            voice_out = gr.Audio(label="🔊 Réponse vocale (dernière)",
                                 autoplay=True, scale=1)

        with gr.Row():
            clear_btn = gr.Button("🗑️ Effacer")
            refresh_btn = gr.Button("🔄 Rafraîchir la base")
            speak_chk = gr.Checkbox(label="🔊 Lire les réponses à voix haute",
                                    value=False)

        # ---- wiring : text AND voice feed the SAME chatbot ----------------
        # Text is TWO phases: add_user_text echoes the question instantly + clears the
        # box, then reply_text generates. Voice always speaks back; typed answers speak
        # only when the toggle is on. Every long-running event is captured so the Stop
        # button can cancel it.
        # show_progress="hidden" removes Gradio's ugly "processing | 55s/32s" queue-ETA
        # boxes that used to clutter the output components; the chat echo + Stop button
        # already make it obvious the bot is working.
        send_evt = send_btn.click(add_user_text, [txt, chatbot], [chatbot, txt],
                                  show_progress="hidden") \
            .then(reply_text, [chatbot, speak_chk], [chatbot, voice_out],
                  show_progress="hidden")
        submit_evt = txt.submit(add_user_text, [txt, chatbot], [chatbot, txt],
                                show_progress="hidden") \
            .then(reply_text, [chatbot, speak_chk], [chatbot, voice_out],
                  show_progress="hidden")
        voice_evt = mic.stop_recording(on_voice, [mic, chatbot], [chatbot, voice_out],
                                       show_progress="hidden")

        stop_btn.click(None, None, None, cancels=[send_evt, submit_evt, voice_evt])
        clear_btn.click(on_clear, None, [chatbot, txt, voice_out])
        refresh_btn.click(on_refresh, None, status)

    return demo


def main():
    boot()
    demo = build_ui()
    # allowed_paths: let Gradio serve the saved TTS wavs so spoken-answer bubbles stay
    # replayable later in the conversation (mic recordings live in Gradio's own cache).
    os.makedirs(voice._tts_dir, exist_ok=True)   # exist before launch references it
    demo.queue(default_concurrency_limit=1).launch(
        share=config.GRADIO_SHARE, allowed_paths=[voice._tts_dir])


if __name__ == "__main__":
    main()
