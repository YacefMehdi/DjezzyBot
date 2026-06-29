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

# A bare arithmetic query ("combien font 24 fois 7 ?", "24 x 7"). The word "combien" is
# a telecom cue (it keeps "combien coûte X" in domain), so without this an arithmetic
# question would skip the OOD gate; we force it through the gate instead.
_ARITH_RE = re.compile(r"\d+\s*(?:fois|x|×|\*|\+|/|plus|moins|divis|multipli|%)\s*\d+",
                       re.IGNORECASE)

# Foreign-script characters that never belong in a Djezzy answer (any of our languages):
# CJK (Chinese/Japanese/Korean) and Cyrillic. Used to build the decode-time token ban below;
# these token IDs are masked during generation so the model can't code-switch into them.
_FOREIGN_SCRIPT_RE = re.compile(r"[一-鿿぀-ヿ가-힯Ѐ-ӿ]")

# Brand-token repair — a cheap safety net AFTER the decode-time ban. With Cyrillic tokens
# forbidden, the mixed-script "دжезzy" can't form; this only catches a residual script-mixed
# brand token (e.g. partial-byte fragments that slipped the ban) and rebuilds the canonical
# spelling. Matches any letter run spanning Latin/Cyrillic/Arabic and rewrites it only when it
# mixes Cyrillic with another script (the corruption signature) — a clean word is untouched.
_LETTER_RUN_RE = re.compile(r"[A-Za-zЀ-ӿ؀-ۿ]{2,}")
_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _repair_brands(text: str, lang: str) -> str:
    """Rebuild a script-mixed (corrupted) 'Djezzy' token to its canonical spelling.

    Touches ONLY tokens that mix Cyrillic with Latin/Arabic (the corruption signature),
    so a clean 'Djezzy'/'جيزي' or any normal word is left untouched. Arabic/Darija answers
    get 'جيزي', Latin-script answers get 'Djezzy'. Microsecond deterministic backstop to the
    decode-time script ban (no second generation involved)."""
    canonical = "جيزي" if lang in ("ar", "dz") else "Djezzy"

    def _fix(m):
        tok = m.group(0)
        if _CYRILLIC_RE.search(tok) and (_LATIN_RE.search(tok) or _ARABIC_RE.search(tok)):
            return canonical
        return tok

    return _LETTER_RUN_RE.sub(_fix, text)


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
    _build_script_ban(_tokenizer)        # one-time: vocab IDs to forbid at decode time
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
    "Tu réponds EXCLUSIVEMENT à partir du CONTEXTE fourni ci-dessous. Les 11 règles suivantes "
    "sont ABSOLUES et prioritaires sur toute autre considération — tu ne les enfreins JAMAIS :\n"
    "1. DOMAINE DJEZZY UNIQUEMENT. Tu ne traites QUE des sujets Djezzy : offres, forfaits, "
    "prix, services, recharge/Flexy, roaming, réseau, carte SIM, téléphonie et internet. Pour "
    "TOUTE question hors de ce domaine (météo, politique, cuisine, sport, blague, calcul ou "
    "mathématiques, géographie, capitale ou nom d'un pays/d'une ville, histoire, science, "
    "heure/date, culture générale, conseils personnels...), tu refuses en UNE phrase polie et tu "
    "réorientes vers Djezzy. INTERDICTION ABSOLUE : ne donne, ne mentionne, ne laisse JAMAIS "
    "transparaître la réponse à une telle question — même si tu la connais, même partiellement, "
    "même « à titre d'information », et dans AUCUNE langue. Exemple : à « quelle est la capitale "
    "de la France ? », tu réponds que tu ne traites que les sujets Djezzy, SANS écrire le nom de "
    "la ville nulle part dans ta réponse.\n"
    "2. ZÉRO INVENTION. N'invente JAMAIS : ni prix, ni offre, ni volume (Go/Mo), ni validité, "
    "ni code USSD, ni numéro, ni condition. Tu ne cites QUE ce qui figure EXPLICITEMENT dans le "
    "contexte. Si l'information demandée n'y est pas, dis-le honnêtement et clairement — ne "
    "devine pas, n'estime pas, ne comble jamais par une valeur « probable » ou « habituelle ».\n"
    "3. PRIX EXACT, AUCUN OUBLI. Chaque offre que tu cites porte son prix EXACT, tel qu'écrit "
    "dans le contexte. Le niveau de détail attendu (bref ou complet) t'est indiqué dans la "
    "« Consigne de présentation » de la question — respecte-le scrupuleusement. Quand le détail "
    "complet est demandé, n'omets AUCUN élément présent dans le contexte : volume internet/5G, "
    "validité ou date limite, appels nationaux, appels vers les autres opérateurs, SMS nationaux "
    "et internationaux, réseaux sociaux inclus, crédit ou appels/SMS offerts, éligibilité "
    "(ex : étudiant), numéro spécial (ex : 0770), code USSD. Si DEUX forfaits ont le MÊME prix "
    "mais un contenu différent, présente-les comme DEUX forfaits distincts — ne les fusionne "
    "jamais en un seul.\n"
    "4. BUDGET. Si un budget est indiqué, n'affiche QUE des offres dont le prix est inférieur "
    "ou égal à ce budget. Le contexte est déjà filtré et contient PLUSIEURS offres éligibles : "
    "présente-les TOUTES, sans en oublier une seule, et ne prétends jamais qu'il n'existe qu'une "
    "seule option quand il y en a plusieurs.\n"
    "5. NATIONAL ≠ ROAMING. Ne confonds JAMAIS un tarif national avec un tarif roaming "
    "(à l'étranger). N'emploie un tarif roaming que si la question porte explicitement sur "
    "l'étranger, et un tarif national que pour l'usage en Algérie.\n"
    "6. AUCUN CONCURRENT (une seule exception). Ne décris, ne compare, ne recommande JAMAIS "
    "les offres d'un autre opérateur (Ooredoo, Mobilis...). SEULE EXCEPTION : tu peux indiquer "
    "qu'une offre Djezzy inclut des appels ou SMS VERS ces réseaux — c'est une caractéristique de "
    "l'offre Djezzy, pas la promotion d'un concurrent.\n"
    "7. LANGUE DU CLIENT. Réponds TOUJOURS et ENTIÈREMENT dans la langue de la question. Ne "
    "mélange jamais deux langues dans une même phrase (hors noms propres et chiffres).\n"
    "8. DARIJA → ARABE STANDARD. Si la question est en darija algérien, réponds en arabe "
    "standard moderne (MSA), clair et correct — jamais en darija écrit.\n"
    "9. NOMS EN LATIN. Conserve les noms d'offres, de services et de destinations dans leur "
    "alphabet latin d'origine (Djezzy, iZZY, Legend, Campuce, Flexy...), même au sein d'une "
    "réponse en arabe ; ne les translittère jamais en caractères arabes.\n"
    "10. PAS DE RÉPÉTITION. Ne répète jamais deux fois la même offre, le même palier ou la "
    "même phrase.\n"
    "11. CLAIR ET CONCIS. Réponds de manière structurée, directe et concise, sans formule de "
    "remplissage ni bavardage — va droit à l'information utile."
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
    "ar": "أجب بالكامل باللغة العربية الفصحى. اكتب أسماء العروض واسم المشغّل بالحروف "
          "اللاتينية كما هي تمامًا (Djezzy, iZZY, Legend, Flexy, Zid, Cam Puce, Confort...) "
          "ولا تنقلها أبدًا إلى الحروف العربية.",
    "dz": "أجب بالكامل باللغة العربية الفصحى (المعيارية)، حتى لو كان السؤال بالدارجة. "
          "اكتب أسماء العروض واسم المشغّل بالحروف اللاتينية كما هي تمامًا "
          "(Djezzy, iZZY, Legend, Flexy, Zid, Cam Puce, Confort...) ولا تنقلها أبدًا "
          "إلى الحروف العربية.",
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
        "Le client indique un budget. Le contexte contient PLUSIEURS offres éligibles "
        "(souvent 3 à 5) : tu DOIS les présenter TOUTES, jamais une seule. Pour CHAQUE "
        "offre du contexte, donne son nom puis, parmi ses paliers au prix inférieur ou "
        "égal au budget, mets en avant le palier le PLUS AVANTAGEUX (le plus cher que le "
        "budget permet, donc le plus de Go) avec son prix et ses détails essentiels. "
        "Classe les offres de la moins chère à la plus chère, UNE par ligne. NE DIS "
        "JAMAIS qu'il n'existe qu'une seule option tant que le contexte en contient "
        "plusieurs, et n'oublie aucune offre présente dans le contexte."
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
            f"offre dont le prix dépasse {budget} DA. "
            f"N'AJOUTE JAMAIS une offre ou un prix qui n'apparaît pas EXPLICITEMENT dans "
            f"le contexte ci-dessus : n'invente pas d'offre « premium » ou plus chère, ne "
            f"la suggère pas en comparaison, ne cite que ce qui est écrit dans le "
            f"contexte.\n\n"
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
# Foreign-script ban (the real fix for Qwen's code-switch). Rather than detect a Chinese/
# Cyrillic answer after the fact and regenerate (a wasted second pass with unpredictable
# latency), we forbid those tokens AT DECODE TIME: a LogitsProcessor sets the logit of every
# CJK/Cyrillic vocabulary token to -inf before each pick, so greedy literally cannot select
# one and is forced to the next-best (correct-script) token. Zero extra latency, still fully
# greedy/deterministic, and the mangled "دжезzy" can't even form. The banned-ID list is built
# once at model load by scanning the tokenizer vocabulary.
_BANNED_SCRIPT_IDS = None      # list[int], populated by _build_script_ban
_SCRIPT_GUARD = None           # cached LogitsProcessorList


def _build_script_ban(tokenizer) -> None:
    """Find every vocab token that decodes to a CJK or Cyrillic character (built once).

    These scripts never legitimately appear in a Djezzy answer in ANY of our languages
    (Arabic, French, English, Darija-as-MSA), so suppressing them is language-independent
    and safe. ~151k tokens scanned at load; the resulting ID list is cached for reuse."""
    global _BANNED_SCRIPT_IDS, _SCRIPT_GUARD
    if _BANNED_SCRIPT_IDS is not None:
        return
    strs = tokenizer.batch_decode([[i] for i in range(len(tokenizer))])
    _BANNED_SCRIPT_IDS = [i for i, s in enumerate(strs) if s and _FOREIGN_SCRIPT_RE.search(s)]
    _SCRIPT_GUARD = None         # rebuilt lazily on first generate (needs torch + device)
    logger.info("script ban: forbidding %d CJK/Cyrillic tokens at decode time",
                len(_BANNED_SCRIPT_IDS))


def _script_guard():
    """The cached LogitsProcessorList that masks the banned tokens (lazy, needs torch)."""
    global _SCRIPT_GUARD
    if _SCRIPT_GUARD is None:
        import torch
        from transformers import LogitsProcessor, LogitsProcessorList

        banned = torch.tensor(_BANNED_SCRIPT_IDS or [], dtype=torch.long)

        class _SuppressScripts(LogitsProcessor):
            def __call__(self, input_ids, scores):
                if banned.numel():
                    scores[:, banned.to(scores.device)] = float("-inf")
                return scores

        _SCRIPT_GUARD = LogitsProcessorList([_SuppressScripts()])
    return _SCRIPT_GUARD


def _generate_n(prompt: str, max_new_tokens: int) -> str:
    """Greedy generation (do_sample=False), capped at `max_new_tokens`.

    CJK/Cyrillic tokens are masked out at decode time (see _build_script_ban), so the
    output can never contain a code-switch to Chinese/Russian — no post-hoc scan or
    regeneration needed."""
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
            logits_processor=_script_guard(),
        )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def _generate(prompt: str) -> str:
    """Greedy generation of a full answer (do_sample=False)."""
    return _generate_n(prompt, config.MAX_NEW_TOKENS)


def _generate_clean(question: str, context: str, lang: str, history: list,
                    budget: int, route: str) -> str:
    """Generate an answer (CJK/Cyrillic already forbidden at decode time), then run the
    microsecond _repair_brands net to canonicalise any brand token that slipped. No second
    pass, no sampling — latency is identical to a plain generation."""
    prompt = build_prompt(question, context, lang, history, budget, route)
    return _repair_brands(_generate(prompt), lang)


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


def _in_domain(question: str, history: list = None) -> bool:
    """True if `question` is about Djezzy / telecom (answer it), False if off-topic.

    Receives the recent HISTORY so an elliptical follow-up ("c'est combien ?" right
    after "Parle-moi de Legend") is judged in context and not wrongly refused — without
    it the gate sees only the bare follow-up and may reject an in-domain question.
    Defaults to True on any ambiguous output, so a real customer is never wrongly
    refused because the classifier hesitated (false-refusals are the costly error)."""
    ctx = ""
    if history:
        lines = []
        for msg in history[-2:]:            # last exchange is enough to resolve ellipsis
            who = "Client" if msg["role"] == "user" else "DjezzyBot"
            lines.append(f"{who} : {msg['content'][:200]}")
        ctx = "Contexte récent de la conversation :\n" + "\n".join(lines) + "\n\n"
    user = (
        f"{ctx}"
        "La DERNIÈRE question du client concerne-t-elle l'opérateur de téléphonie "
        "algérien Djezzy — ses offres, forfaits, prix, recharge/Flexy, roaming, réseau, "
        "internet, carte SIM, ou un autre service télécom ?\n"
        "Réponds OUI si le sujet est Djezzy/télécom — Y COMPRIS un suivi comme « c'est "
        "combien ? » ou « et en arabe ? » qui se réfère à une offre déjà évoquée ci-dessus. "
        "Réponds NON seulement si la question est clairement hors sujet (météo, géographie, "
        "capitale, calcul, histoire, sport, blague, culture générale...).\n\n"
        f"Dernière question : {question}\n\nRéponse (un seul mot, OUI ou NON) :"
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

    # OOD domain gate (optional second layer): only the catch-all NORMAL route can be
    # off-topic — the other routes matched a concrete Djezzy cue (offer / budget / roaming).
    # DISABLED by default (config.OOD_GATE_ENABLED): it lifted offline OOD recall to 1.0
    # but false-refused in-domain Arabic/Darija questions in live use, and turning a real
    # customer away is the worse error here. Rule 1 + the similarity floor stay in charge.
    if route == "normal" and config.OOD_GATE_ENABLED:
        # Run the gate ONLY on signal-less queries: anything carrying a telecom / offer /
        # device cue (has_telecom_signal) is trusted in-domain and skips the gate, so a
        # real question is never refused for lacking a recognised phrase. An explicit
        # arithmetic query is gated even though "combien" counts as a cue.
        skip_gate = lexicon.has_telecom_signal(question) and not _ARITH_RE.search(question)
        if not skip_gate:
            with timed(stages, "gate"):
                in_domain = _in_domain(question, history)
            if not in_domain:
                return {"text": _OOD_REFUSAL.get(lang, _OOD_REFUSAL["fr"]),
                        "route": "out_of_domain",
                        "t_retrieval": stages["retrieval"], "t_generation": 0.0,
                        "t_gate": stages["gate"]}

    budget = budget_of(question)        # hard ceiling, non-None only on the budget route
    context = _format_context(docs)
    with timed(stages, "generation"):
        text = _generate_clean(question, context, lang, history, budget, route)

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
