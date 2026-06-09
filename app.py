"""
app.py — Gradio front-end (text + voice) and application wiring.

Boots the bot: loads the FAISS index (building it from the cached scrape if
needed), starts the daily-refresh scheduler, and serves a two-tab Gradio UI in
Djezzy red/white.

    Tab 1 — Text chat : chatbot + send + clear + refresh
    Tab 2 — Voice     : mic -> transcribe (shown) -> answer -> autoplayed speech

A status bar shows the last refresh time and the indexed document count. Language
is auto-detected from the input (no manual selector needed). Launches with
share=True for Colab.

Run:  python app.py
"""

import logging
from datetime import datetime, timezone

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
def _history_to_messages(chat_history):
    """Convert Gradio 'messages' history to the bot's role/content list."""
    msgs = []
    for m in chat_history or []:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
            msgs.append({"role": m["role"], "content": m["content"]})
    return msgs


def on_text_send(message, chat_history):
    """Tab 1: answer a typed message and append to the chat."""
    chat_history = chat_history or []
    if not message or not message.strip():
        return chat_history, ""
    if STATE["index"] is None:
        chat_history += [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ La base n'est pas encore indexée. "
                                             "Cliquez sur « Rafraîchir »."},
        ]
        return chat_history, ""
    prior = _history_to_messages(chat_history)
    result = bot.answer(message, STATE["index"], prior)
    chat_history += [
        {"role": "user", "content": message},
        {"role": "assistant", "content": result["text"]},
    ]
    return chat_history, ""


def on_clear():
    """Tab 1: clear the conversation."""
    return [], ""


def on_refresh():
    """Refresh button: re-scrape + reindex, then update STATE + status bar."""
    result = scheduler.force_refresh()
    if result.get("ok"):
        STATE["index"] = scheduler.CURRENT_INDEX
        STATE["doc_count"] = _index_size(STATE["index"])
        STATE["last_refresh"] = result.get("ts", "")
    return _status_text()


def on_voice(audio_path, chat_history):
    """Tab 2: transcribe -> answer -> speak. Returns transcription, chat, audio."""
    chat_history = chat_history or []
    if audio_path is None:
        return "", chat_history, None
    if STATE["index"] is None:
        return "⚠️ Base non indexée.", chat_history, None
    prior = _history_to_messages(chat_history)
    result = voice.voice_answer(audio_path, STATE["index"], prior)
    transcription = f"🗣️ {result['transcription']}  ({result['lang']})"
    chat_history += [
        {"role": "user", "content": result["transcription"]},
        {"role": "assistant", "content": result["text"]},
    ]
    return transcription, chat_history, result["wav_path"]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
_CSS = f"""
.gradio-container {{ font-family: 'Segoe UI', sans-serif; }}
#title {{ color: {config.DJEZZY_RED}; font-weight: 800; }}
.djezzy-status {{ background: {config.DJEZZY_RED}; color: {config.DJEZZY_WHITE};
                  padding: 8px 14px; border-radius: 8px; font-weight: 600; }}
footer {{ visibility: hidden; }}
"""


def build_ui():
    """Construct and return the Gradio Blocks app (not launched)."""
    import gradio as gr

    theme = gr.themes.Soft(primary_hue=gr.themes.colors.red,
                          neutral_hue=gr.themes.colors.gray)

    with gr.Blocks(theme=theme, css=_CSS, title="DjezzyBot") as demo:
        gr.Markdown("# 📱 DjezzyBot", elem_id="title")
        gr.Markdown("Assistant virtuel multilingue de Djezzy — texte & voix "
                    "(Arabe · Français · English · Darija).")
        status = gr.Markdown(_status_text(), elem_classes=["djezzy-status"])

        with gr.Tabs():
            # ---- Tab 1: text ------------------------------------------------
            with gr.Tab("💬 Chat texte"):
                chatbot = gr.Chatbot(type="messages", height=460, label="Conversation")
                with gr.Row():
                    txt = gr.Textbox(placeholder="Posez votre question à Djezzy…",
                                     scale=8, show_label=False, autofocus=True)
                    send_btn = gr.Button("Envoyer", variant="primary", scale=1)
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Effacer")
                    refresh_btn = gr.Button("🔄 Rafraîchir la base")

            # ---- Tab 2: voice -----------------------------------------------
            with gr.Tab("🎙️ Voix"):
                mic = gr.Audio(sources=["microphone"], type="filepath",
                               label="Parlez à DjezzyBot")
                transcription = gr.Markdown("", label="Transcription")
                voice_out = gr.Audio(label="Réponse vocale", autoplay=True)
                voice_chat = gr.Chatbot(type="messages", height=340, label="Conversation")

        # ---- wiring -------------------------------------------------------
        send_btn.click(on_text_send, [txt, chatbot], [chatbot, txt])
        txt.submit(on_text_send, [txt, chatbot], [chatbot, txt])
        clear_btn.click(on_clear, None, [chatbot, txt])
        refresh_btn.click(on_refresh, None, status)
        mic.stop_recording(on_voice, [mic, voice_chat],
                           [transcription, voice_chat, voice_out])

    return demo


def main():
    boot()
    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(share=config.GRADIO_SHARE)


if __name__ == "__main__":
    main()
