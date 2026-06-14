"""
bot.py — LLM, prompt assembly, language detection, history, and the text path.

Loads Qwen2.5-7B (4-bit NF4), assembles the Qwen chat prompt (system rules +
retrieved context + recent history + a per-language directive placed in the USER
turn, never the assistant turn), generates greedily, and returns the answer with
per-stage latency already measured.

Public API
----------
    load_llm()                                   -> (model, tokenizer)
    detect_language(text)                        -> "fr" | "ar" | "en" | "dz"
    answer(question, vector_db, history)         -> dict
    LATENCY                                       -> accumulated timing store

`answer()` returns:
    {text, lang, route, t_retrieval, t_generation}
and appends a record to the module-level LATENCY store so REPORT.md can be built
from real measurements rather than estimates.

Latency is wired in from the start (per spec): retrieval and generation are timed
SEPARATELY via the `timed()` context manager.
"""

import re
import time
import json
import logging
from contextlib import contextmanager

import config
from data import lexicon
from retriever import smart_retrieve, budget_of, classify_route

logger = logging.getLogger("djezzybot.bot")

_model = None
_tokenizer = None

# Arabic letters (used by language detection).
_ARABIC_RE = re.compile(r"[؀-ۿ]")


# ===========================================================================
# Latency store (shared across the whole test run)
# ===========================================================================
# Each record: {"kind": "text"|"voice", "route": str, "stages": {name: seconds}}
LATENCY = []


@contextmanager
def timed(bucket: dict, name: str):
    """Measure wall-clock seconds for a stage and store it in `bucket[name]`."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        bucket[name] = time.perf_counter() - t0


def record_latency(kind: str, route: str, stages: dict):
    """Append a latency record and persist the store to disk (best-effort)."""
    LATENCY.append({"kind": kind, "route": route, "stages": stages})
    try:
        with open(config.LATENCY_STORE, "w", encoding="utf-8") as f:
            json.dump(LATENCY, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


# ===========================================================================
# Model loading
# ===========================================================================
def load_llm():
    """Load Qwen2.5-7B-Instruct (4-bit NF4) and its tokenizer (cached)."""
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info("loading LLM %s", config.LLM_MODEL_ID)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    _tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_ID)
    _model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL_ID,
        quantization_config=bnb,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    _model.eval()
    # Qwen's generation_config ships max_length=32768, which clashes with our
    # max_new_tokens=768 and makes transformers warn on every generate(). Drop it
    # so only max_new_tokens governs the cap (behaviour unchanged, warning gone).
    _model.generation_config.max_length = None
    return _model, _tokenizer


# ===========================================================================
# Language detection
# ===========================================================================
def detect_language(text: str) -> str:
    """Classify into fr / ar / en / dz.

    Order matters: Darija markers are checked first (they're written in Latin
    script and would otherwise be mistaken for French), then Arabic script, then
    langdetect distinguishes French vs English (defaulting to French on failure).
    """
    t = text.lower()
    tokens = set(re.findall(r"[a-z0-9]+", t))
    if tokens & lexicon.DARIJA_WORDS:
        return "dz"
    if _ARABIC_RE.search(text):
        return "ar"
    try:
        from langdetect import detect
        lang = detect(text)
    except Exception:
        lang = "fr"
    return "en" if lang == "en" else "fr"


# ===========================================================================
# System prompt — 10 strict rules
# ===========================================================================
SYSTEM_PROMPT = (
    "Tu es DjezzyBot, l'assistant virtuel officiel de l'opérateur télécom algérien Djezzy. "
    "Tu réponds uniquement à partir du CONTEXTE fourni. Respecte ces 11 règles ABSOLUES :\n"
    "1. DOMAINE DJEZZY UNIQUEMENT : tu ne réponds QU'aux questions portant sur Djezzy — ses "
    "offres, forfaits, prix, services, recharge, roaming, réseau, téléphonie et internet. Si la "
    "question sort de ce domaine (météo, politique, cuisine, blagues, sport, calcul ou "
    "mathématiques, géographie, capitale ou nom d'un pays/d'une ville, histoire, science, heure "
    "ou date, culture générale, conseils personnels...), tu REFUSES poliment en une seule phrase "
    "et tu réorientes vers les offres Djezzy — MÊME si tu connais parfaitement la réponse (par "
    "exemple la capitale d'un pays ou le résultat d'un calcul), tu ne la donnes JAMAIS, dans "
    "AUCUNE langue.\n"
    "2. CONTEXTE UNIQUEMENT : n'invente JAMAIS un prix, une offre, un volume, une validité "
    "ou un code USSD. Si l'information demandée n'est pas dans le contexte, dis-le honnêtement "
    "plutôt que de deviner.\n"
    "3. PRIX EXACT, RIEN D'INVENTÉ : chaque offre que tu cites doit porter son prix exact. Le "
    "NIVEAU DE DÉTAIL attendu (bref ou complet) t'est indiqué dans la « Consigne de présentation » "
    "de la question — respecte-le. Quand le détail complet est demandé, n'oublie AUCUN élément "
    "présent dans le contexte : volume internet / 5G, validité ou date limite, appels nationaux, "
    "appels vers les autres opérateurs, SMS nationaux, SMS internationaux, réseaux sociaux inclus, "
    "crédit ou appels/SMS entrants offerts, conditions d'éligibilité (ex : étudiant), numéro "
    "spécial (ex : 0770), code USSD. N'invente jamais : ne cite que ce qui est dans le contexte.\n"
    "4. BUDGET : si un budget est donné, n'affiche QUE des offres dont le prix est inférieur "
    "ou égal à ce budget. Le contexte est déjà filtré — liste tout ce qu'il contient.\n"
    "5. NATIONAL ≠ ROAMING : ne confonds jamais un tarif national avec un tarif roaming "
    "(à l'étranger). N'utilise un tarif roaming que si la question concerne l'étranger.\n"
    "6. PAS DE CONCURRENT (avec une exception) : ne décris, ne compare et ne recommande jamais "
    "les OFFRES d'un autre opérateur. EXCEPTION : tu PEUX indiquer qu'une offre Djezzy inclut des "
    "appels ou SMS VERS d'autres réseaux (Ooredoo, Mobilis...) — c'est une caractéristique de "
    "l'offre Djezzy, pas une promotion d'un concurrent.\n"
    "7. LANGUE DU CLIENT : réponds TOUJOURS dans la langue de la question.\n"
    "8. DARIJA → ARABE STANDARD : si la question est en darija algérien, réponds en arabe "
    "standard moderne (MSA), clair et correct.\n"
    "9. NOMS LATINS : conserve les noms d'offres et de destinations en alphabet latin "
    "d'origine (Legend, iZZY, Campuce...), même dans une réponse en arabe.\n"
    "10. PAS DE RÉPÉTITION : ne répète pas deux fois la même offre ou la même phrase.\n"
    "11. CONCIS : réponds de manière claire, structurée et concise, sans bavardage."
)

# Canned competitor refusal, per language.
_COMPETITOR_REFUSAL = {
    "fr": "Je suis l'assistant virtuel de Djezzy et je ne peux pas vous renseigner sur "
          "les offres d'autres opérateurs. Je serai ravi de vous présenter les offres Djezzy.",
    "en": "I am Djezzy's virtual assistant and cannot provide information about other "
          "operators. I'd be glad to tell you about Djezzy's offers.",
    "ar": "أنا المساعد الافتراضي لجيزي ولا يمكنني تقديم معلومات عن المشغّلين الآخرين. "
          "يسعدني أن أعرّفك بعروض جيزي.",
    "dz": "أنا المساعد الافتراضي لجيزي ولا يمكنني تقديم معلومات عن المشغّلين الآخرين. "
          "يسعدني أن أعرّفك بعروض جيزي.",
}

# Empty-context fallback, per language (no chunks matched).
_NO_CONTEXT = {
    "fr": "Je ne trouve pas cette information dans la base Djezzy actuelle. "
          "Pouvez-vous reformuler ou préciser l'offre qui vous intéresse ?",
    "en": "I can't find this in the current Djezzy data. Could you rephrase or specify "
          "which offer you mean?",
    "ar": "لا أجد هذه المعلومة في قاعدة بيانات جيزي الحالية. هل يمكنك إعادة الصياغة أو "
          "تحديد العرض الذي يهمّك؟",
    "dz": "لا أجد هذه المعلومة في قاعدة بيانات جيزي الحالية. هل يمكنك تحديد العرض الذي يهمّك؟",
}

# Per-language reply directive — placed in the USER turn (a known fix: directives
# in the assistant turn leak into the output).
_LANG_DIRECTIVE = {
    "fr": "Réponds entièrement en français.",
    "en": "Reply entirely in English.",
    "ar": "أجب بالكامل باللغة العربية الفصحى.",
    "dz": "أجب بالكامل باللغة العربية الفصحى (المعيارية)، حتى لو كان السؤال بالدارجة.",
}

# Per-INTENT presentation directive — chosen from the route, so the answer's level
# of detail and ordering follow what the user actually asked. This is what stops the
# bot from dumping every detail on "vos offres" or listing offers in a random order:
# the same intent that picked the retrieval strategy now also dictates the format.
_STYLE_DIRECTIVE = {
    "catalogue": (
        "Le client veut un APERÇU de tes gammes. Présente CHAQUE gamme du contexte sur "
        "UNE seule ligne : nom + prix de départ (« à partir de X DA ») + 3 mots "
        "d'accroche au maximum. NE DÉTAILLE PAS les forfaits ici. N'OMETS AUCUNE gamme "
        "présente dans le contexte, et GARDE l'ordre du contexte (déjà classé du moins "
        "cher au plus cher). Termine en invitant le client à demander une gamme précise "
        "pour en avoir le détail complet."
    ),
    "named": (
        "Le client veut une offre précise. Donne son prix, puis TOUS ses forfaits/paliers "
        "et TOUS les détails présents dans le contexte, sans en oublier un seul. S'il y a "
        "plusieurs paliers, liste-les du moins cher au plus cher."
    ),
    "comparison": (
        "Le client compare des offres. Présente-les l'une après l'autre, chacune avec son "
        "prix et ses caractéristiques clés, puis résume en une phrase la différence. Classe "
        "de la moins chère à la plus chère."
    ),
    "budget": (
        "Liste les offres éligibles, UNE par ligne, de la moins chère à la plus chère, "
        "chacune avec son prix et ses détails essentiels."
    ),
    "roaming": (
        "Donne les tarifs/forfaits roaming (à l'étranger) demandés, avec leur prix ; ne les "
        "confonds jamais avec un tarif national."
    ),
    "normal": "Réponds précisément à la question, uniquement à partir du contexte.",
}


# ===========================================================================
# History (manual sliding window — NOT LangChain memory)
# ===========================================================================
def _format_history(history: list) -> str:
    """Render the last HISTORY_WINDOW messages as a short transcript block.

    `history` is a plain list of {"role": "user"|"assistant", "content": str}.
    We take history[-HISTORY_WINDOW:] (last 4 exchanges) so long chats don't blow
    up the prompt or lose the thread.
    """
    window = history[-config.HISTORY_WINDOW:] if history else []
    if not window:
        return ""
    lines = ["Historique récent :"]
    for msg in window:
        who = "Client" if msg["role"] == "user" else "DjezzyBot"
        lines.append(f"- {who} : {msg['content']}")
    return "\n".join(lines) + "\n"


# ===========================================================================
# Prompt assembly
# ===========================================================================
def _format_context(docs: list) -> str:
    """Join retrieved chunks into a context block, capped at MAX_CONTEXT_CHARS."""
    parts = []
    total = 0
    for d in docs:
        src = d.metadata.get("source_url", "")
        block = f"[Source: {src}]\n{d.page_content}"
        if total + len(block) > config.MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def build_prompt(question: str, context: str, lang: str, history: list,
                 budget: int = None, route: str = "normal") -> str:
    """Assemble the full Qwen chat-ML prompt string.

    Structure: system rules → (user turn) history + context + [budget note] +
    presentation directive (chosen from `route`) + language directive + question →
    empty assistant turn. All instructions live in the system or USER turn; the
    assistant turn is left empty so the model only produces the answer.

    `route` selects the presentation style (catalogue = brief menu, named = full
    detail, comparison = side-by-side, budget = cheapest-first, …). When `budget`
    is set, a hard ceiling is also stated (the context is already Python-filtered
    to <=budget, but this makes the model enforce it too, and clarifies that
    crédit/bonus amounts may exceed the budget).
    """
    hist = _format_history(history)
    directive = _LANG_DIRECTIVE.get(lang, _LANG_DIRECTIVE["fr"])
    style = _STYLE_DIRECTIVE.get(route, _STYLE_DIRECTIVE["normal"])
    budget_note = ""
    if budget is not None:
        budget_note = (
            f"BUDGET DU CLIENT : {budget} DA. N'affiche AUCUNE offre dont le PRIX "
            f"dépasse {budget} DA. Le contexte ne contient que des offres éligibles. "
            f"Attention : les montants de crédit/bonus inclus dans une offre peuvent "
            f"dépasser {budget} DA — ce n'est pas le prix, ne les confonds pas. "
            f"Si AUCUNE offre du contexte ne coûte {budget} DA ou moins, dis clairement "
            f"qu'aucune offre n'est disponible à ce budget et invite le client à "
            f"augmenter son budget — ne propose JAMAIS, même à titre indicatif, une "
            f"offre dont le prix dépasse {budget} DA.\n\n"
        )
    user_turn = (
        f"{hist}"
        f"Contexte (base de données Djezzy) :\n{context if context else '(aucun)'}\n\n"
        f"{budget_note}"
        f"Consigne de présentation : {style}\n\n"
        f"{directive}\n\n"
        f"Question du client : {question}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_turn},
    ]
    # Use the tokenizer's chat template when available (correct Qwen formatting),
    # otherwise fall back to a manual ChatML string (keeps pure tests working).
    if _tokenizer is not None:
        return _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_turn}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ===========================================================================
# Generation
# ===========================================================================
def _generate_n(prompt: str, max_new_tokens: int) -> str:
    """Greedy generation (do_sample=False), capped at `max_new_tokens`."""
    import torch
    model, tokenizer = load_llm()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=config.DO_SAMPLE,
            repetition_penalty=config.REPETITION_PENALTY,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def _generate(prompt: str) -> str:
    """Greedy generation of a full answer (do_sample=False)."""
    return _generate_n(prompt, config.MAX_NEW_TOKENS)


# ===========================================================================
# Out-of-domain DOMAIN GATE — a dedicated binary classifier (LLM-as-judge)
# ===========================================================================
# Why a separate call instead of trusting rule 1: in the full generation the model
# juggles 11 rules + retrieved context that LOOKS usable + formatting + language, so
# the one-line refusal rule gets crowded out and OOD questions slip through (measured
# OOD recall ~0.20 with the inline rule alone). A FOCUSED yes/no prompt — no context,
# no other rules, nothing to format — turns refusal into a pure classification, which
# the same model does far more reliably. This is the second layer of the OOD guard
# (the first is retriever._is_out_of_domain's similarity floor); only the catch-all
# NORMAL route is gated (named/budget/roaming/… already matched a Djezzy cue).
_GATE_SYSTEM = (
    "Tu es un classifieur binaire. Tu réponds EXCLUSIVEMENT par un seul mot : "
    "OUI ou NON. Aucune autre sortie n'est autorisée."
)


def _in_domain(question: str) -> bool:
    """True if `question` is about Djezzy / telecom (answer it), False if off-topic.

    Defaults to True on any ambiguous output, so a real customer is never wrongly
    refused because the classifier hesitated (false-refusals are the costly error)."""
    user = (
        "La question suivante d'un client concerne-t-elle l'opérateur de téléphonie "
        "algérien Djezzy — ses offres, forfaits, prix, recharge/Flexy, roaming, réseau, "
        "internet, carte SIM, ou un autre service télécom ?\n"
        "Réponds OUI si le sujet est Djezzy/télécom, NON s'il est hors sujet (météo, "
        "géographie, capitale, calcul, histoire, sport, blague, culture générale...).\n\n"
        f"Question : {question}\n\nRéponse (un seul mot, OUI ou NON) :"
    )
    messages = [{"role": "system", "content": _GATE_SYSTEM},
                {"role": "user", "content": user}]
    if _tokenizer is not None:
        prompt = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = (f"<|im_start|>system\n{_GATE_SYSTEM}<|im_end|>\n"
                  f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
    verdict = _generate_n(prompt, 5).strip().lower()
    return not (verdict.startswith("non") or verdict.startswith("no") or "لا" in verdict)


# Out-of-domain refusal, per language (polite, redirects to Djezzy).
_OOD_REFUSAL = {
    "fr": "Je suis l'assistant virtuel de Djezzy et je ne réponds qu'aux questions "
          "concernant Djezzy (offres, forfaits, recharge, roaming, réseau...). "
          "Comment puis-je vous aider à ce sujet ?",
    "en": "I'm Djezzy's virtual assistant and I only answer questions about Djezzy "
          "(offers, plans, recharge, roaming, network...). How can I help you with that?",
    "ar": "أنا المساعد الافتراضي لجيزي وأجيب فقط عن الأسئلة المتعلقة بجيزي (العروض، "
          "الباقات، التعبئة، التجوال، الشبكة...). كيف يمكنني مساعدتك في هذا المجال؟",
    "dz": "أنا المساعد الافتراضي لجيزي وأجيب فقط عن الأسئلة المتعلقة بجيزي (العروض، "
          "الباقات، التعبئة، التجوال، الشبكة...). كيف يمكنني مساعدتك في هذا المجال؟",
}


# ===========================================================================
# Shared core (used by BOTH the text path and the voice path)
# ===========================================================================
def generate_answer(question: str, lang: str, vector_db, history: list = None) -> dict:
    """Retrieve → route handling → build prompt → generate. The single source of
    truth for "understand the question and answer it", so the text path (answer)
    and the voice path (voice.voice_answer) can never drift.

    Returns {text, route, t_retrieval, t_generation}. Retrieval and generation are
    timed separately; competitor/empty(out-of-domain) routes skip generation and
    return a canned reply. `lang` is supplied by the caller (detect_language for
    text, the STT-detected language for voice).
    """
    history = history or []
    stages = {}

    route = classify_route(question)    # the intent — drives retrieval AND presentation
    with timed(stages, "retrieval"):
        docs = smart_retrieve(question, lang, vector_db)

    # competitor firewall / out-of-domain / no match — no generation needed
    if docs == config.COMPETITOR_SENTINEL:
        return {"text": _COMPETITOR_REFUSAL.get(lang, _COMPETITOR_REFUSAL["fr"]),
                "route": "competitor",
                "t_retrieval": stages["retrieval"], "t_generation": 0.0}
    if not docs:
        return {"text": _NO_CONTEXT.get(lang, _NO_CONTEXT["fr"]),
                "route": "no_context",
                "t_retrieval": stages["retrieval"], "t_generation": 0.0}

    # OOD domain gate (second layer): only the catch-all NORMAL route can be off-topic
    # — the other routes matched a concrete Djezzy cue (offer name / budget / roaming).
    # A focused yes/no classification refuses what the inline rule let slip, without a
    # full generation. Timed separately so its cost shows up honestly in the latency.
    if route == "normal":
        with timed(stages, "gate"):
            in_domain = _in_domain(question)
        if not in_domain:
            return {"text": _OOD_REFUSAL.get(lang, _OOD_REFUSAL["fr"]),
                    "route": "out_of_domain",
                    "t_retrieval": stages["retrieval"], "t_generation": 0.0,
                    "t_gate": stages["gate"]}

    budget = budget_of(question)        # hard ceiling, non-None only on the budget route
    context = _format_context(docs)
    prompt = build_prompt(question, context, lang, history, budget, route)
    with timed(stages, "generation"):
        text = _generate(prompt)

    return {"text": text, "route": route,
            "t_retrieval": stages["retrieval"], "t_generation": stages["generation"]}


# ===========================================================================
# Public: text answer
# ===========================================================================
def answer(question: str, vector_db, history: list = None) -> dict:
    """Answer one typed question end-to-end. Thin wrapper over generate_answer
    that detects the language and records text-path latency. Returns
    {text, lang, route, t_retrieval, t_generation}.
    """
    lang = detect_language(question)
    res = generate_answer(question, lang, vector_db, history)
    record_latency("text", res["route"],
                   {"retrieval": res["t_retrieval"], "generation": res["t_generation"]})
    return {"text": res["text"], "lang": lang, "route": res["route"],
            "t_retrieval": res["t_retrieval"], "t_generation": res["t_generation"]}
