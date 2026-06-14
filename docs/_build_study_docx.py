# -*- coding: utf-8 -*-
"""Builds the DjezzyBot defense study guide as a .docx (python-docx)."""
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

RED = RGBColor(0xE4, 0x00, 0x2B)
doc = Document()

# base font
st = doc.styles["Normal"]
st.font.name = "Calibri"; st.font.size = Pt(11)

def h1(t):
    p = doc.add_heading(t, level=1)
    for r in p.runs: r.font.color.rgb = RED
def h2(t):
    doc.add_heading(t, level=2)
def para(t, italic=False, bold=False):
    p = doc.add_paragraph()
    r = p.add_run(t); r.italic = italic; r.bold = bold
    return p
def qa(n, q, a):
    p = doc.add_paragraph()
    r = p.add_run("Q%d.  %s" % (n, q)); r.bold = True
    ap = doc.add_paragraph(a)
    ap.paragraph_format.space_after = Pt(8)

# ---- Title ----
t = doc.add_heading("DjezzyBot — Master's Defense Study Guide", level=0)
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub = para("Question-and-answer guide, organised by study session. Updated with the "
           "evaluation metrics, related work, ethics and claim-framing added in review.")
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
for r in sub.runs: r.italic = True

# ---- How to use ----
h1("How to use this guide")
para("The jury grades whether you UNDERSTAND the thesis, not the code. ~70% of questions "
     "target your own contributions (Sessions 2-3). Both authors must own those; split the "
     "background (Sessions 1 and 5).")
para("For every card: (1) read the question, (2) close the guide and say the answer OUT LOUD "
     "in your own words, (3) only then check the answer below it, (4) mark what you fumbled and "
     "restudy just that. Passive reading does not stick — explaining aloud does.", italic=True)
para("Pair this with the flashcards (docs/defense_prep.html) for the self-test step, and the "
     "thesis itself as the source of truth.")

# =====================================================================
SESSIONS = [

("SESSION 1 — Foundations & Vocabulary (do first; it makes everything else click)", [
 ("What is NLP and how did it evolve?",
  "Natural Language Processing = getting computers to understand and produce human language. "
  "It went: hand-written rules -> statistical machine learning (HMM/SVM) -> deep learning "
  "(RNN/LSTM) -> today's Transformers and LLMs. Our project sits at the LLM end of that line."),
 ("Why did Transformers replace RNNs/LSTMs?",
  "RNNs read word-by-word, which makes them slow (no parallelism), prone to forgetting "
  "long-range information (the vanishing-gradient problem), and unable to use future context. "
  "Self-attention looks at all words at once and fixes all three."),
 ("Explain self-attention / Q-K-V simply.",
  "Each word sends a Query; every word offers a Key (what it is about) and a Value (its meaning). "
  "Attention matches queries to keys -- softmax(Q.K^T / sqrt(d_k)) -- and blends the values. "
  "That is how the model gets context: 'Legend' becomes the plan, not a hero, from its neighbours."),
 ("Encoder vs decoder; BERT vs GPT?",
  "Encoder-only (BERT) is built for UNDERSTANDING (classification, extraction). Decoder-only (GPT) "
  "is built for GENERATION, predicting each next word from the previous ones. Modern LLMs like Qwen "
  "are the decoder line -- that is what we use to write answers."),
 ("What is an LLM and how is it trained?",
  "A Transformer with billions of parameters. Two stages: pre-training (predict the next word on "
  "huge text, which teaches grammar + broad knowledge) then instruction-tuning / RLHF (learn to "
  "follow human instructions). We use Qwen2.5-7B-Instruct."),
 ("What is tokenization / BPE?",
  "Models read tokens (word-pieces turned into numbers), not letters. Byte-Pair Encoding builds the "
  "vocabulary by repeatedly merging the most frequent character pairs. Qwen's vocabulary is ~151k "
  "tokens. Tokenization fixes vocabulary size, speed, and how cleanly a language is handled."),
 ("What is an embedding?",
  "A piece of text turned into a vector of numbers that captures its meaning -- similar meanings land "
  "as nearby vectors. We use multilingual-e5 so French, Arabic, English and Darija share ONE meaning "
  "space, which is why we need no translation step."),
 ("What is cosine similarity?",
  "It measures how close two meanings are by the ANGLE between their vectors: 1 = same meaning, "
  "0 = unrelated. Because our e5 vectors are L2-normalized, cosine equals the inner product FAISS "
  "computes."),
 ("What is FAISS and why IndexFlatIP (exact)?",
  "FAISS is a fast vector-search library. IndexFlatIP is exact brute-force inner product = exact "
  "cosine on normalized vectors. Our corpus is small (~769 chunks) so exact search is instant; "
  "approximate indexes (IVF) only matter at millions of vectors."),
 ("What is RAG and why use it?",
  "Retrieval-Augmented Generation: retrieve the real text first, paste it into the prompt, and the "
  "LLM answers ONLY from it. Two payoffs: no hallucinated facts, and instant updates (change the data, "
  "no retraining). The bot never answers from the model's own memory."),
 ("What is quantization / NF4 and what do you lose?",
  "NF4 stores each weight in 4 bits instead of 16, shrinking Qwen-7B from ~14 GB to ~5 GB so it fits "
  "the free T4, with little quality loss. The cost: the bitsandbytes 4-bit kernel is memory-saving but "
  "slow (~10-15 tokens/second) -- the reason local latency is high."),
 ("What is greedy decoding (do_sample=False)?",
  "At each step pick the single most-likely token, no randomness. In a factual RAG setting 'creativity' "
  "becomes hallucination, so greedy is safer; it is also reproducible, which lets the acceptance tests "
  "assert exact behaviour."),
 ("What is ChatML, and where do the instructions go?",
  "Qwen's prompt format with system / user / assistant roles. We put ALL instructions in the system and "
  "USER turns and leave the assistant turn empty -- a deliberate fix, because directives placed in the "
  "assistant turn leak into the output."),
 ("What does LangChain do here? And Gradio?",
  "LangChain glues the RAG steps (load -> chunk -> embed -> index -> search -> build prompt); it is "
  "convenience, not the intelligence. Gradio is the web UI -- a single shared screen where text and "
  "voice feed one conversation."),
]),

("SESSION 2 — Your Contributions I: Pipeline & Router (master cold)", [
 ("Walk me through the system end-to-end when a user asks a question.",
  "OFFLINE (indexing): scrape Djezzy -> clean HTML -> chunk 1200/120 -> embed with multilingual-e5 "
  "(passage: prefix) -> FAISS IndexFlatIP. ONLINE (inference): detect language -> the intent router "
  "picks 1 of 7 routes -> retrieve chunks (the route decides how) -> build the prompt (system rules + "
  "retrieved context, all in the user turn) -> Qwen2.5-7B greedy-generates -> answer in the user's "
  "language -> (voice) XTTS speaks it. Text and voice call the SAME generate_answer."),
 ("What does the 7-route intent router add over plain vector search?",
  "Pure dense search broke five ways: coined names, comparison imbalance, vague catalogue questions, "
  "budget/roaming confusion, and competitors. A rule-based router reads the question BEFORE FAISS and "
  "picks the right strategy. It is NOT a hybrid retriever -- no BM25, no reranker; the search stays "
  "pure dense FAISS, the router only decides HOW to query it."),
 ("List the 7 routes in priority order -- why that order?",
  "competitor > roaming > catalogue > comparison > budget > named > normal. Most-specific first, so an "
  "ambiguous question falls to the rule that best fits. Two subtleties: catalogue YIELDS to budget (an "
  "explicit amount means 'filter', not 'list all'); and a 2-offer comparison BEATS budget."),
 ("Why does the embedding model miss 'Campuce'?",
  "Campuce is an invented brand (Campus + puce), in no training corpus, so e5 cannot retrieve it by "
  "meaning. The router instead walks the chunk text and matches the literal offer name (whole-word, "
  "accent-insensitive). Fine-tuning could not learn a never-seen word; the router can."),
 ("Why sort offers cheapest-first in Python instead of asking the LLM?",
  "So the answer never depends on the LLM obeying 'sort these'. One Python helper splits an offer's "
  "text into tier blocks (skipping per-unit rates like '5 DA/SMS' and credit/bonus amounts) and orders "
  "them ascending -- the context arrives already sorted. This killed two bugs: jumbled tiers and "
  "'dumps everything'."),
 ("How do you tell 'offre Ooredoo' from 'iZZY calls Mobilis'?",
  "The competitor firewall refuses only when a rival is the SUBJECT. If a call-destination cue ('vers', "
  "'calls to') sits just before the brand, it is a Djezzy offer feature -> allowed through. It is pure "
  "Python so it is instant -- a fast, reliable first line, but NOT infallible: a misspelling like "
  "'Oredoo' slips past and falls through to the LLM's no-competitor rule (the backstop)."),
 ("Why no reranker / how is the context assembled?",
  "On a corpus this small the router already selects the right chunks and orders the priced ones, so a "
  "cross-encoder reranker would add latency and a model to load for little gain. The selected chunks "
  "are concatenated, each tagged [Source: url], capped at 8000 characters; because they arrive ordered, "
  "truncation only drops the least relevant tail."),
]),

("SESSION 3 — Your Contributions II: Multilingual, OOD, Evaluation (the bits the jury probes hardest)", [
 ("How do you support Darija without a translator?",
  "No translation layer. multilingual-e5 embeds fr/ar/en/Darija in one shared space, so a Darija "
  "question matches Arabic/French pages directly; a one-line directive tells Qwen which language to "
  "answer in (Darija -> Modern Standard Arabic). An early Opus-MT pivot corrupted brand names and "
  "prices (it read 'Legend' as the Arabic word for 'recruitment'), so we dropped translation entirely."),
 ("How does language detection work?",
  "Three priority layers: (1) Darija markers first -- including Arabizi spellings with digits like "
  "'ch7al', 'bghit' -- because langdetect mistakes Darija for French; (2) Arabic script; (3) langdetect "
  "for French vs English. Measured at 0.977 accuracy on the balanced evaluation set."),
 ("Why can't a similarity threshold detect out-of-domain questions? (your research finding)",
  "We measured ~95 queries in all four languages: e5 scores EVERYTHING high and the bands overlap -- "
  "real customers 0.774-0.846, off-topic noise 0.748-0.833, pure gibberish 0.798-0.838 (gibberish "
  "sometimes scores HIGHER than a real question). So a score threshold cannot separate them. We use two "
  "layers: a very low floor (0.70) that only catches degenerate input, and the LLM's domain rule that "
  "judges by MEANING. The trade: we never refuse a real customer, especially a short Darija question."),
 ("How did you evaluate the system?",
  "Two parts. (1) A behavioral ACCEPTANCE suite -- 14 deterministic pass/fail scenarios -- which is "
  "functional validation, NOT an accuracy metric. (2) A quantitative metric evaluation (evaluate.py): "
  "per-class precision/recall/F1 + confusion for language detection and intent routing, retrieval "
  "recall@k versus a no-router baseline, and an automatic answer-groundedness score. Labels are "
  "objective by construction, so there is no self-grading bias."),
 ("Is 14/14 your accuracy?",
  "No -- and we say so explicitly. 14/14 is a behavioral acceptance suite (does the system do the "
  "specified right thing on each case), made deterministic by greedy decoding. The real accuracy figures "
  "come from the metric-based evaluation (precision/recall/recall@k against a baseline)."),
 ("What are your actual numbers?",
  "Language detection: accuracy 0.977, macro-F1 0.976. Intent routing: accuracy 1.0, macro-F1 1.0 across "
  "all 7 routes and 4 languages. Retrieval recall@k (router vs baseline), end-to-end OOD F1, and "
  "groundedness come from the Colab run. We keep a sub-100% number on purpose -- it shows the evaluation "
  "is honest, not rigged."),
 ("What baseline do you compare against, and why?",
  "Pure dense retrieval with no router and no lexicon expansion. We compare its recall@k to the router's "
  "recall@k to quantify exactly what the router contributes. That is the ablation the supervisor asked "
  "for."),
 ("How do you measure hallucination objectively?",
  "Groundedness: for every answer, each price it states must appear in the retrieved context. The "
  "fraction of grounded prices is an automatic, objective anti-hallucination score -- no human judgment "
  "needed."),
 ("How do you guarantee voice and text answer identically?",
  "Both paths call the same generate_answer -- 'one brain'. The voice path just adds Whisper before and "
  "XTTS after; the retrieval, routing and generation in between are identical, so they cannot drift."),
]),

("SESSION 4 — Stack, Latency & Fine-tuning (be fluent, one-line 'why' per choice)", [
 ("Why Qwen2.5-7B, not an Arabic model (Jais/Falcon)?",
  "Qwen2.5 has excellent Arabic + French, a 128K context, strong instruction-following, and at 7B it "
  "fits a free T4 with 4-bit quantization. Falcon/Jais are heavier or weaker at multilingual "
  "instruction-following in this weight class."),
 ("Why multilingual-e5 as the embedder?",
  "One model embeds all four languages in the same space -> no translation needed. A monolingual English "
  "embedder could not handle half our traffic (Arabic/Darija)."),
 ("Why local instead of an API -- isn't the API faster?",
  "We started on Groq (Llama-3, cloud): fast, but rate-limit failures and subscriber data leaving the "
  "country. We moved to a 100% local open-source stack for data sovereignty and no vendor lock-in. "
  "Slower on a free T4, but the architecture is identical."),
 ("Isn't it slow? (latency -- know this cold)",
  "On the free T4 + 4-bit, a long answer takes tens of seconds -- that is the HARDWARE, not the "
  "architecture. Our own Groq prototype on the SAME pipeline answered in ~1s (text) / ~2-2.5s (voice). "
  "Retrieval is under 2s; generation is ~99% of the time and scales with answer length. Production GPUs "
  "(A100/H100) restore real-time speed."),
 ("Why is the Groq-vs-Colab comparison only qualitative?",
  "Because it confounds model AND infrastructure (Llama-3 on a Groq ASIC vs Qwen-7B 4-bit on a T4) -- "
  "they vary together, so it isolates neither. We draw no model-quality conclusion from the speed gap; "
  "a controlled study would fix the model and vary only the hardware."),
 ("Why RAG instead of fine-tuning Qwen on Djezzy data?",
  "Fine-tuning changes behaviour/form; RAG supplies facts. Prices change daily, so FT would bake stale "
  "facts into the weights and need constant retraining; there is no labelled Darija set; a free T4 "
  "cannot train 7B; and FT does NOT stop hallucination. RAG keeps facts current and grounded."),
 ("Did you fine-tune anything? How would you, if forced?",
  "No -- RAG for knowledge, an 11-rule prompt for behaviour, an STT brand-primer, and XTTS zero-shot "
  "cloning: all inference-time adaptation. If forced: QLoRA on the LLM for dialect TONE; contrastive "
  "fine-tuning of e5 on synthetic in-domain pairs; Whisper on Algerian audio; a Piper voice on Algerian "
  "Arabic. All PEFT, all future work -- for FORM, never for facts."),
 ("Why is the embedder the most justifiable fine-tune -- and why didn't you?",
  "Because it directly governs retrieval quality. We chose a cheaper deterministic alternative for two "
  "reasons: (1) it needs labelled (query, passage) pairs we'd have to synthesize, and (2) it STILL could "
  "not retrieve a coined brand it never saw (the Campuce cold-start). A router with exact lexical match "
  "solves Campuce that a fine-tuned retriever cannot. That is engineering judgement, not laziness."),
]),

("SESSION 5 — Context (headline awareness only -- don't over-study)", [
 ("Give the Algerian telecom market figures.",
  "~54 million mobile subscribers (ARPCE Q4-2024), ~117% penetration. Three operators: Mobilis (state), "
  "Ooredoo (Qatar), Djezzy. Djezzy launched 2002, ~14 million subscribers, owned FNI 51% / VEON 49%."),
 ("What competitor virtual assistants exist, and their limits for Algeria?",
  "TOBi (Vodafone), Djingo (Orange), Google Duplex/Contact Center AI. They perform well but: no Darija "
  "support, high licensing costs, and dependence on foreign cloud infrastructure -- which is the gap "
  "DjezzyBot fills as an open-source, sovereign, multilingual alternative."),
 ("What is the open-source LLM landscape, and why Qwen?",
  "Llama 3, Mistral/Mixtral, Falcon, Bloom, Qwen 2.5. We picked Qwen: best Arabic/multilingual support, "
  "128K context, and a 7B size that fits the free T4 at 4-bit."),
 ("What academic work exists on Algerian/Maghrebi dialectal NLP?",
  "NArabizi treebank (romanized Algerian UGC, Seddah 2020); DziriBERT (first Algerian-dialect "
  "Transformer, Abdaoui 2021); DarijaBERT (Moroccan, Gaanoun 2024); DZDC12 (Algerian Arabizi-French "
  "corpus, Abainia 2019); TUNIZI (Tunisian Arabizi sentiment, Fourati 2020). They are mostly encoders or "
  "corpora for classification -- none is a deployed multilingual telecom RAG assistant. That is our gap."),
 ("Could you have used DziriBERT for retrieval?",
  "It could sharpen retrieval, but it is monolingual-by-dialect -- on its own it would not place a Darija "
  "question and a French offer page in ONE shared space the way multilingual-e5 does, which our "
  "no-translation design depends on. It is a future fine-tuning avenue for the embedder."),
 ("Is scraping djezzy.dz legal and ethical?",
  "We read only PUBLIC pages, collect NO personal data, and crawl politely (browser User-Agent, "
  "randomized delay, rate cap). And the work is a formal Projet de Fin d'Etudes internship hosted by "
  "Djezzy under a stage convention (executive decree 13-306), in their Data Science & Advanced Analytics "
  "department -- so Djezzy is the documented host organization, not an unwitting third party."),
 ("Which requirements matter most?",
  "7 functional + 6 non-functional. Know NFR-02 (latency target -- a PRODUCTION goal, met by the API path "
  "and expected on production GPUs, not the free-T4 prototype) and NFR-03 (fit in 15 GB -> 4-bit "
  "quantization)."),
]),

("SESSION 6 — Live-demo checklist (not Q&A -- rehearse the demo)", [
 ("What must the live demo show, and how do you de-risk it?",
  "Pre-load the Colab BEFORE you present (it takes minutes to boot the models!). Keep a BACKUP screen "
  "recording in case the network or Colab fails. Show: one answer per language (fr/ar/en/Darija), an "
  "out-of-domain refusal (the weather), a competitor refusal (Ooredoo), a budget query, and one voice "
  "round-trip. Have the questions written down so you don't fumble live."),
]),
]

for title, cards in SESSIONS:
    h1(title)
    for i, (q, a) in enumerate(cards, 1):
        qa(i, q, a)

# ---- Fine-tuning table ----
h1("Reference table — Fine-tuning, per model (the jury WILL probe this)")
para("One principle answers 90% of it: fine-tuning changes a model's behaviour/form; RAG supplies "
     "knowledge/facts. For any 'why didn't you fine-tune X?': would it buy form or facts, and do we have "
     "labelled data + compute?", italic=True)
ft = doc.add_table(rows=1, cols=4); ft.style = "Light Grid Accent 1"
hdr = ft.rows[0].cells
for c, txt in zip(hdr, ["Model", "FT could give", "Why we didn't", "What we did instead"]):
    c.paragraphs[0].add_run(txt).bold = True
rows = [
 ("LLM (Qwen-7B)", "tone, baked-in catalogue, Darija",
  "prices change daily -> FT bakes in stale facts; no labelled Darija; T4 can't train 7B; FT doesn't stop hallucination",
  "RAG + 11-rule prompt + greedy"),
 ("Embedder (e5)", "better domain retrieval",
  "needs labelled (query,passage) pairs; STILL can't learn a never-seen coined word (Campuce)",
  "router + exact-name match (deterministic, cheaper)"),
 ("STT (Whisper)", "Algerian dialect + brand terms",
  "needs transcribed Algerian audio -- none available",
  "initial_prompt brand primer (zero training)"),
 ("TTS (XTTS-v2)", "Algerian accent / persona",
  "XTTS already does zero-shot voice cloning",
  "built-in / cloned voice; Piper for production"),
]
for a, b, c, d in rows:
    cells = ft.add_row().cells
    cells[0].text = a; cells[1].text = b; cells[2].text = c; cells[3].text = d
para("Vocabulary to drop: PEFT / LoRA / QLoRA (train tiny adapters, feasible on small GPUs); "
     "catastrophic forgetting (narrow FT degrades general ability); contrastive learning (how you'd FT "
     "the embedder -- pull a query and its relevant passage together).")

# ---- OOD calibration table ----
h1("Reference table — Off-domain calibration (own this; it is your research finding)")
para("On the real index, ~95 queries across 4 languages: the cosine-score bands OVERLAP, so a threshold "
     "cannot separate real from off-topic. Relevance is therefore judged by the LLM's meaning, not a score.")
ct = doc.add_table(rows=1, cols=2); ct.style = "Light Grid Accent 1"
ch = ct.rows[0].cells
ch[0].paragraphs[0].add_run("Query group").bold = True
ch[1].paragraphs[0].add_run("Best-chunk cosine score").bold = True
for grp, rng in [("Real customer questions", "0.774 - 0.846"),
                 ("Off-topic noise (jokes, weather...)", "0.748 - 0.833"),
                 ("Pure gibberish (keyboard mash)", "0.798 - 0.838  (sometimes HIGHER than real)")]:
    r = ct.add_row().cells; r[0].text = grp; r[1].text = rng

# ---- Question bank ----
h1("Rehearsal — Question bank (drill ALOUD, in random order, no notes)")
bank = [
 "Walk me through the system end-to-end when I type a question.",
 "Why RAG and not fine-tuning? Why not just use ChatGPT?",
 "How do you prevent invented prices?",
 "What does the intent router add over plain vector search?",
 "Why does the embedding model miss 'Campuce'?",
 "How do you support Darija without a translator?",
 "Why can't a similarity threshold detect off-domain questions?",
 "Why sort offers in Python instead of asking the LLM?",
 "Why Qwen, not an Arabic-specific model?",
 "What is 4-bit quantization and what do you lose?",
 "Why FAISS IndexFlatIP, not an approximate index?",
 "Why greedy decoding?",
 "What do the passage:/query: prefixes do?",
 "Why these chunk sizes (1200/120)?",
 "What happens when Djezzy changes a price?",
 "Your budget route is the weakest -- why, and how would you fix it?",
 "How do you tell a competitor question from 'iZZY calls Mobilis'?",
 "How did you evaluate? Is 14/14 your accuracy?",
 "What real metrics do you report, and what baseline?",
 "How do you guarantee voice and text answer identically?",
 "Explain attention / Q-K-V simply.",
 "Difference between your router and a 'hybrid retriever'?",
 "Why local over an API -- isn't the API faster?",
 "What does 'data sovereignty' mean and why does it matter?",
 "What are the system's main limitations and your future work?",
]
for i, q in enumerate(bank, 1):
    p = doc.add_paragraph(); p.add_run("%d. " % i).bold = True; p.add_run(q)

# ---- Traps & one-liners ----
h1("Traps & one-liners (composure wins points)")
for t in [
 "'Wouldn't fine-tuning be more accurate?' -> For FACTS, no -- worse: stale prices + hallucination. "
 "RAG keeps facts current. FT only helps FORM, which wasn't our bottleneck.",
 "'An API is faster.' -> Yes, that's the hardware. Our API prototype on the SAME pipeline answered in "
 "~1 second; production GPUs will too. We used a free T4 to prove the open-source stack works at zero cost.",
 "'Is 14/14 not too good?' -> It's a behavioral acceptance suite, not an accuracy metric -- deterministic "
 "pass/fail. The real metrics (precision/recall/recall@k vs a baseline) are reported separately.",
 "For any gotcha you don't know: 'I'm not certain, but here is how I'd find out.' Composure beats bluffing.",
]:
    p = doc.add_paragraph(t, style="List Bullet")

doc.save("DjezzyBot_Defense_Study_Guide.docx")
print("saved DjezzyBot_Defense_Study_Guide.docx")
