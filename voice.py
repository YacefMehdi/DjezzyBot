"""
voice.py — Whisper STT + Coqui XTTS-v2 TTS, with voice round-trip latency.

The voice path: microphone WAV -> Whisper transcription (auto language detect) ->
bot.generate_answer (the SAME shared brain the text path uses) -> XTTS-v2 speech.
XTTS is loaded ON DEMAND (first synth call) to keep VRAM free while only typing.

Public API
----------
    load_stt()                         -> whisper model (cached)
    transcribe(audio_path)             -> (text, lang)
    load_tts()                         -> XTTS model (cached, lazy)
    synthesize(text, lang)             -> wav_path
    voice_answer(audio_path, vector_db, history) -> dict

`voice_answer()` returns
    {transcription, lang, text, route, wav_path,
     t_stt, t_retrieval, t_generation, t_tts}
and records the four-stage round trip into bot.LATENCY (kind="voice"), so the
report's voice numbers are real measurements.

Number/abbreviation expansion before TTS (Go, SMS, DA, thousands separators) keeps
the spoken output natural — XTTS reads "5 Go" as words, not letters.
"""

import os
import re
import time
import logging

import config
from bot import generate_answer, detect_language, timed, record_latency

logger = logging.getLogger("djezzybot.voice")

# Coqui XTTS-v2 ships under the CPML license and, the first time the model loads, asks
# the user to accept it via input(). In a non-interactive Colab/Gradio process there is
# no stdin, so that prompt raised "EOFError: EOF when reading a line" and killed every
# voice request. Pre-accepting via this env var (set BEFORE TTS is imported) skips the
# prompt. The licence permits this research / non-commercial use; a production build
# would switch to a permissively-licensed voice such as Piper.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

_stt = None
_tts = None
_tts_dir = os.path.join(config.BASE_DIR, "tts_out")


# ===========================================================================
# STT — Whisper medium
# ===========================================================================
def load_stt():
    """Load (once) Whisper-medium via faster-whisper (CTranslate2).

    float16 on GPU keeps the weights at ~1.5 GB instead of openai-whisper's fp32
    ~3 GB, and is faster — meaningful headroom next to Qwen-7B + XTTS on a T4.
    Falls back to int8 on CPU.
    """
    global _stt
    if _stt is None:
        from faster_whisper import WhisperModel
        import torch
        # config holds the HF-style id ("openai/whisper-medium"); faster-whisper
        # wants the size name ("medium").
        size = config.STT_MODEL_ID.split("whisper-")[-1]
        if torch.cuda.is_available():
            device, compute = "cuda", config.STT_COMPUTE_TYPE
        else:
            device, compute = "cpu", "int8"
        logger.info("loading faster-whisper %s (%s, %s)", size, device, compute)
        _stt = WhisperModel(size, device=device, compute_type=compute)
    return _stt


# Brand-primed prompt to bias Whisper toward Djezzy vocabulary. It also lists common
# Algerian-Darija question words (شحال = how much, كيفاش = how, قداش, رصيد = credit,
# تعبئة = top-up, فليكسي = Flexy) so the model is less likely to mishear them — e.g.
# "شحال عندي كريدي" (how much credit do I have) was being heard as "حال عندي كريدي".
# This is a primer, not a fix for Whisper's limited Darija coverage (Future Work:
# fine-tune on an Algerian-Darija speech corpus).
_STT_PRIMER = (
    "Djezzy, iZZY, Legend, Campuce, Zid, Confort, roaming, forfait, "
    "Go, Mo, DA, dinars, internet, crédit, Hadj, Omra. "
    "شحال، قداش، كيفاش، رصيد، كريدي، تعبئة، فليكسي، عرض، باقة."
)


_STT_OPTS = dict(task="transcribe", initial_prompt=_STT_PRIMER,
                 temperature=0.0, beam_size=3, condition_on_previous_text=False)


def transcribe(audio_path: str):
    """Transcribe audio and return (text, lang) with lang in fr/ar/en/dz.

    Language policy (mechanism: one allowed-language set, enforced at the source):
    Whisper auto-detects first, but if it guesses a language we DON'T support (e.g.
    German for a French question), we re-transcribe while FORCING the most probable
    supported language. That fixes the actual transcription text, not just its label
    — a wrong-language guess used to produce meaningless text the bot couldn't
    answer. Darija is heard as Arabic, then upgraded to "dz" by the text detector.
    """
    model = load_stt()
    allowed = set(config.STT_ALLOWED_LANGS)

    segments, info = model.transcribe(audio_path, **_STT_OPTS)
    if info.language not in allowed:
        # pick the best language AMONG the ones we support, then transcribe as that
        probs = dict(getattr(info, "all_language_probs", None) or [])
        forced = max(allowed, key=lambda l: probs.get(l, 0.0)) if probs else "fr"
        logger.info("STT detected unsupported '%s' → forcing '%s'", info.language, forced)
        segments, info = model.transcribe(audio_path, language=forced, **_STT_OPTS)

    text = "".join(seg.text for seg in segments).strip()
    wlang = info.language if info.language in allowed else "fr"
    # reuse the text-side detector so STT and text paths agree on dz vs ar/fr
    lang = detect_language(text) if text else wlang
    # if Whisper heard Arabic but our detector didn't catch Darija, keep ar
    if wlang == "ar" and lang == "fr":
        lang = "ar"
    return text, lang


# ===========================================================================
# TTS — Coqui XTTS-v2 (on-demand)
# ===========================================================================
def load_tts():
    """Load (once, lazily) Coqui XTTS-v2. Called on the first synth, not at boot."""
    global _tts
    if _tts is None:
        import torch
        # Coqui-TTS's tortoise layer does `from transformers.pytorch_utils import
        # isin_mps_friendly`, which newer transformers no longer expose — the import
        # then dies with ImportError. Provide the function if it's missing (it's just
        # a thin wrapper over torch.isin), so TTS imports across transformers versions.
        import transformers.pytorch_utils as _ptu
        if not hasattr(_ptu, "isin_mps_friendly"):
            _ptu.isin_mps_friendly = lambda elements, test_elements: torch.isin(elements, test_elements)
        os.environ.setdefault("COQUI_TOS_AGREED", "1")   # skip the interactive CPML prompt
        from TTS.api import TTS
        logger.info("loading XTTS-v2 (on demand) %s", config.TTS_MODEL_ID)
        _tts = TTS(config.TTS_MODEL_ID).to("cuda" if torch.cuda.is_available() else "cpu")
    return _tts


# Number-to-words + abbreviation expansion so XTTS speaks naturally.
_TTS_NUM_LANG = {"fr": "fr", "en": "en", "ar": "ar", "dz": "ar"}


def _expand_for_tts(text: str, lang: str) -> str:
    """Turn written telecom text into naturally SPEAKABLE text (one layer, by class).

    A voice assistant must never spell things out wrong, so we normalize whole
    CATEGORIES rather than single cases:
    - thousands separators: "3 000" -> "3000" (so it's read "trois mille")
    - units: Go / Mo / SMS / DA -> spoken words (language-appropriate)
    - USSD / short codes: "*123#" -> symbols + digits spoken one by one
    - special phone-style numbers (leading zero, e.g. "0770") -> digit by digit
    - any remaining integer -> words via num2words
    """
    from num2words import num2words

    nlang = _TTS_NUM_LANG.get(lang, "fr")

    # 1) collapse "1 000" / "3 000" style separators BEFORE word conversion
    text = re.sub(r"(?<=\d)\s+(?=\d{3}\b)", "", text)

    # 2) telecom units -> spoken words (+ symbol names for the USSD pass)
    if lang in ("ar", "dz"):
        repl = {"Go": "جيجابايت", "Mo": "ميجابايت", "SMS": "رسائل",
                "DA": "دينار", "DZD": "دينار"}
        star, hashm = "نجمة", "مربع"
    elif lang == "en":
        repl = {"Go": "gigabytes", "Mo": "megabytes", "SMS": "SMS",
                "DA": "dinars", "DZD": "dinars"}
        star, hashm = "star", "hash"
    else:  # fr
        repl = {"Go": "giga-octets", "Mo": "méga-octets", "SMS": "SMS",
                "DA": "dinars", "DZD": "dinars"}
        star, hashm = "étoile", "dièse"
    for k, v in repl.items():
        text = re.sub(rf"\b{k}\b", v, text)

    def _digit(ch):
        try:
            return num2words(int(ch), lang=nlang)
        except Exception:
            return ch

    def _say_digits(digits):
        return " ".join(_digit(ch) for ch in digits)

    # 3) USSD / short codes ("*123#", "#100*1#") -> symbols + digits, one by one
    def _ussd(m):
        out = []
        for ch in m.group(0):
            if ch == "*":
                out.append(star)
            elif ch == "#":
                out.append(hashm)
            elif ch.isdigit():
                out.append(_digit(ch))
        return " ".join(out)
    text = re.sub(r"[*#][\d*#]*\d[\d*#]*#?|\b\d+#", _ussd, text)

    # 4) special phone-style numbers (leading zero, e.g. 0770) -> digit by digit
    text = re.sub(r"\b0\d{2,}\b", lambda m: _say_digits(m.group(0)), text)

    # 5) remaining plain integers -> words
    def _num(m):
        try:
            return num2words(int(m.group(0)), lang=nlang)
        except Exception:
            return m.group(0)
    return re.sub(r"\d+", _num, text)


def synthesize(text: str, lang: str) -> str:
    """Synthesize `text` to a WAV file and return its path.

    XTTS only has fr/en/ar voices, so Darija ("dz") is spoken with the Arabic
    voice via config.TTS_LANG_MAP. Text is expanded for natural pronunciation.
    """
    model = load_tts()
    os.makedirs(_tts_dir, exist_ok=True)
    xtts_lang = config.TTS_LANG_MAP.get(lang, "fr")
    spoken = _expand_for_tts(text, lang)
    out_path = os.path.join(_tts_dir, f"tts_{int(time.time()*1000)}.wav")
    # XTTS-v2 is a multi-speaker (voice-cloning) model; newer Coqui builds REQUIRE
    # an explicit speaker. Use the first built-in studio voice so the call never
    # errors and every answer keeps the same consistent voice. (Single-speaker
    # builds expose no `speakers`, so we simply omit it and keep the old behaviour.)
    kwargs = {}
    speakers = getattr(model, "speakers", None) or []
    if speakers:
        kwargs["speaker"] = speakers[0]
    model.tts_to_file(text=spoken, language=xtts_lang, file_path=out_path, **kwargs)
    return out_path


# ===========================================================================
# Full voice round trip
# ===========================================================================
def voice_answer(audio_path: str, vector_db, history: list = None) -> dict:
    """STT -> retrieve -> generate -> TTS, timing each stage separately.

    Returns transcription, detected language, answer text, route, the spoken WAV
    path, and the four stage timings. Records a kind="voice" latency entry.
    """
    history = history or []
    stages = {}

    with timed(stages, "stt"):
        transcription, lang = transcribe(audio_path)

    # Shared brain: identical retrieval/routing/budget/prompt/generation as text.
    res = generate_answer(transcription, lang, vector_db, history)
    stages["retrieval"] = res["t_retrieval"]
    stages["generation"] = res["t_generation"]
    text, route = res["text"], res["route"]

    with timed(stages, "tts"):
        wav_path = synthesize(text, lang)

    record_latency("voice", route, stages)
    return {
        "transcription": transcription, "lang": lang, "text": text, "route": route,
        "wav_path": wav_path,
        "t_stt": stages.get("stt", 0.0),
        "t_retrieval": stages.get("retrieval", 0.0),
        "t_generation": stages.get("generation", 0.0),
        "t_tts": stages.get("tts", 0.0),
    }
