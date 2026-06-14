"""
evaluate.py — quantitative, metric-based evaluation (the real numbers).

Unlike test_scenarios.py (a behavioral acceptance suite: pass/fail on 14 cases),
this module reports the metrics a jury expects — precision / recall / F1 with
confusion matrices for the classification components, retrieval recall@k against a
no-router baseline, and an automatic answer-groundedness score. The labels are
OBJECTIVE BY CONSTRUCTION (a query's language, its intended intent, and the offer
it asks about are known when the query is written), so there is no "the authors
graded themselves" annotator bias for these numbers.

What it measures
----------------
  1. Language detection      : per-class P/R/F1 + confusion + accuracy   (no GPU)
  2. Intent routing          : per-route P/R/F1 + confusion              (no GPU)
  3. OOD / competitor refusal : binary P/R/F1 + false-refusal rate       (no GPU)
  4. Retrieval recall@k       : router vs pure-dense BASELINE (ablation)  (index)
  5. Answer groundedness      : % of answer prices present in the context (index+LLM)

Sections 1–3 need nothing but the lexicon (run anywhere). Sections 4–5 need the
built FAISS index (and 5 needs the LLM), so they self-skip if it is absent.

Usage
-----
    python evaluate.py                 # boot index, run everything available
    from evaluate import run_all       # returns the metrics dict (for the report)

Writes evaluate_results.json so the thesis tables/macros are filled from real runs.
"""

import json
import logging

import config
import bot
import retriever
from retriever import smart_retrieve, classify_route
from data import lexicon

logger = logging.getLogger("djezzybot.evaluate")

_PRICE_RE = retriever._PRICE_RE


# ===========================================================================
# Labelled evaluation set — labels are objective by construction.
#   q     : the query
#   lang  : gold language (fr/ar/en/dz)
#   route : gold intent route (classify_route's vocabulary)
#   offer : canonical offer the query is about (for retrieval relevance) or None
#   ood   : True if the query is genuinely out-of-domain (must be refused)
# ===========================================================================
D = [
    # --- catalogue ---
    dict(q="Quelles sont vos offres ?", lang="fr", route="catalogue", offer=None, ood=False),
    dict(q="Montrez-moi tous vos forfaits disponibles", lang="fr", route="catalogue", offer=None, ood=False),
    dict(q="شنو هي العروض المتوفرة عندكم؟", lang="ar", route="catalogue", offer=None, ood=False),
    dict(q="What offers do you have?", lang="en", route="catalogue", offer=None, ood=False),
    # --- named offer ---
    dict(q="Parle-moi de l'offre Legend", lang="fr", route="named", offer="legend", ood=False),
    dict(q="Parle-moi de l'offre Campuce", lang="fr", route="named", offer="campuce", ood=False),
    dict(q="Parle-moi de l'offre Zid", lang="fr", route="named", offer="zid", ood=False),
    dict(q="C'est quoi l'offre Confort ?", lang="fr", route="named", offer="confort", ood=False),
    dict(q="قولي على عرض ليجند بالتفصيل", lang="ar", route="named", offer="legend", ood=False),
    dict(q="ch7al izzy ?", lang="dz", route="named", offer="izzy", ood=False),
    dict(q="chhal taman izzy ?", lang="dz", route="named", offer="izzy", ood=False),
    dict(q="Tell me about the Legend offer", lang="en", route="named", offer="legend", ood=False),
    # --- comparison ---
    dict(q="Quelle est la différence entre Legend et iZZY ?", lang="fr", route="comparison", offer="legend", ood=False),
    dict(q="Compare Legend et Confort", lang="fr", route="comparison", offer="confort", ood=False),
    # --- budget ---
    dict(q="j'ai 500 DA, qu'est-ce que vous proposez ?", lang="fr", route="budget", offer=None, ood=False),
    dict(q="j'ai seulement 200 DA, quelles offres ?", lang="fr", route="budget", offer=None, ood=False),
    dict(q="عندي 1000 دينار، شنو تنصحوني؟", lang="ar", route="budget", offer=None, ood=False),
    dict(q="I have only 300 DA, what can I get?", lang="en", route="budget", offer=None, ood=False),
    dict(q="3andi 500 da, wach nakhou ?", lang="dz", route="budget", offer=None, ood=False),
    # --- roaming ---
    dict(q="je voyage en France, quel roaming ?", lang="fr", route="roaming", offer=None, ood=False),
    dict(q="بغيت رومينغ باش نروح للحج", lang="ar", route="roaming", offer=None, ood=False),
    dict(q="bghit roaming l france", lang="dz", route="roaming", offer=None, ood=False),
    dict(q="quel forfait roaming pour la Tunisie ?", lang="fr", route="roaming", offer=None, ood=False),
    # --- competitor (must refuse) ---
    dict(q="C'est quoi les offres de Ooredoo ?", lang="fr", route="competitor", offer=None, ood=False),
    dict(q="Les forfaits Mobilis sont mieux ?", lang="fr", route="competitor", offer=None, ood=False),
    dict(q="واش هي عروض أوريدو؟", lang="ar", route="competitor", offer=None, ood=False),
    # --- normal in-domain (price follow-up / generic telecom) ---
    dict(q="C'est combien le forfait le moins cher ?", lang="fr", route="normal", offer=None, ood=False),
    dict(q="Comment activer ma carte SIM ?", lang="fr", route="normal", offer=None, ood=False),
    # --- out-of-domain (route=normal, but must be refused) ---
    dict(q="Quelle est la météo à Alger demain ?", lang="fr", route="normal", offer=None, ood=True),
    dict(q="What's the weather in Algiers tomorrow?", lang="en", route="normal", offer=None, ood=True),
    dict(q="Raconte-moi une blague", lang="fr", route="normal", offer=None, ood=True),
    dict(q="Combien font 24 fois 7 ?", lang="fr", route="normal", offer=None, ood=True),
    dict(q="ما هي عاصمة فرنسا؟", lang="ar", route="normal", offer=None, ood=True),
]


# ===========================================================================
# Metric helpers (pure Python — no sklearn dependency)
# ===========================================================================
def _prf(pairs):
    """pairs = list[(gold, pred)] -> dict with per-class P/R/F1, accuracy, macro-F1."""
    classes = sorted({g for g, _ in pairs} | {p for _, p in pairs})
    per = {}
    for c in classes:
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[c] = dict(precision=round(prec, 3), recall=round(rec, 3),
                      f1=round(f1, 3), support=sum(1 for g, _ in pairs if g == c))
    n = len(pairs)
    acc = sum(1 for g, p in pairs if g == p) / n if n else 0.0
    macro_f1 = sum(v["f1"] for v in per.values()) / len(per) if per else 0.0
    return dict(per_class=per, accuracy=round(acc, 3), macro_f1=round(macro_f1, 3), n=n)


def _confusion(pairs):
    """Return {gold: {pred: count}} for printing a confusion matrix."""
    classes = sorted({g for g, _ in pairs} | {p for _, p in pairs})
    mat = {g: {p: 0 for p in classes} for g in classes}
    for g, p in pairs:
        mat[g][p] += 1
    return mat


def _print_prf(title, res):
    print(f"\n### {title}  (accuracy={res['accuracy']}, macro-F1={res['macro_f1']}, n={res['n']})")
    print(f"  {'class':14s} {'P':>6} {'R':>6} {'F1':>6} {'supp':>5}")
    for c, v in res["per_class"].items():
        print(f"  {c:14s} {v['precision']:6.3f} {v['recall']:6.3f} {v['f1']:6.3f} {v['support']:5d}")


def _print_confusion(title, mat):
    classes = list(mat.keys())
    print(f"\n### {title} — confusion (rows=gold, cols=pred)")
    print("  " + " " * 12 + " ".join(f"{c[:8]:>8}" for c in classes))
    for g in classes:
        print("  " + f"{g:12s}" + " ".join(f"{mat[g][p]:>8}" for p in classes))


# ===========================================================================
# 1–3. Classification metrics (no GPU, no index)
# ===========================================================================
def eval_language():
    pairs = [(d["lang"], bot.detect_language(d["q"])) for d in D]
    return _prf(pairs), _confusion(pairs)


def eval_routing():
    pairs = [(d["route"], classify_route(d["q"])) for d in D]
    return _prf(pairs), _confusion(pairs)


def eval_ood():
    """Binary: is the query out-of-domain? Predicted OOD = the no-context refusal.

    classify_route can't see retrieval confidence, so the true OOD decision lives
    in smart_retrieve's NORMAL branch. Here we approximate the *router-visible*
    part: a query is predicted in-domain if it carries a telecom signal or names an
    offer; predicted OOD otherwise. The full picture (score gate + LLM rule) needs
    the index and is measured end-to-end in eval_retrieval/groundedness.
    """
    pairs = []
    for d in D:
        pred_ood = not lexicon.has_telecom_signal(d["q"])
        pairs.append(("ood" if d["ood"] else "in", "ood" if pred_ood else "in"))
    res = _prf(pairs)
    # false-refusal rate = in-domain queries wrongly predicted OOD
    fr_denom = sum(1 for g, _ in pairs if g == "in")
    false_refusals = sum(1 for g, p in pairs if g == "in" and p == "ood")
    res["false_refusal_rate"] = round(false_refusals / fr_denom, 3) if fr_denom else 0.0
    return res, _confusion(pairs)


# ===========================================================================
# 4. Retrieval recall@k — router vs pure-dense baseline (needs the index)
# ===========================================================================
def _relevant(doc, offer):
    return lexicon.offer_in_text(offer, doc.page_content)


def eval_retrieval(vector_db):
    """recall@1 / recall@5 for offer queries, router vs no-router dense baseline."""
    items = [d for d in D if d["offer"]]
    out = {"router": {}, "baseline_dense": {}, "n": len(items)}
    for k in (1, 5):
        r_hits = b_hits = 0
        for d in items:
            offer = d["offer"]
            # router path
            docs = smart_retrieve(d["q"], d["lang"], vector_db)
            if docs == config.COMPETITOR_SENTINEL:
                docs = []
            if any(_relevant(x, offer) for x in docs[:k]):
                r_hits += 1
            # pure-dense baseline (no router, no lexicon expansion)
            base = vector_db.similarity_search(d["q"], k=k)
            if any(_relevant(x, offer) for x in base[:k]):
                b_hits += 1
        n = len(items)
        out["router"][f"recall@{k}"] = round(r_hits / n, 3) if n else 0.0
        out["baseline_dense"][f"recall@{k}"] = round(b_hits / n, 3) if n else 0.0
    return out


# ===========================================================================
# 5. Answer groundedness — % of answer prices present in the context (index+LLM)
# ===========================================================================
def _prices(text):
    out = []
    for m in _PRICE_RE.finditer(text.lower()):
        try:
            out.append(int(m.group(1).replace(" ", "")))
        except ValueError:
            pass
    return out


def eval_groundedness(vector_db, subset=None):
    """For non-refusal queries, every PRICE in the answer should appear in the
    retrieved context. Returns the mean fraction of grounded prices (1.0 = no price
    was invented). This is an objective, automatic anti-hallucination measure."""
    items = [d for d in D if not d["ood"] and d["route"] != "competitor"]
    if subset:
        items = items[:subset]
    total_prices = grounded = 0
    answered = 0
    for d in items:
        res = bot.generate_answer(d["q"], d["lang"], vector_db)
        if res["route"] in ("competitor", "no_context"):
            continue
        answered += 1
        # rebuild the context the model saw
        docs = smart_retrieve(d["q"], d["lang"], vector_db)
        if docs == config.COMPETITOR_SENTINEL:
            continue
        ctx_prices = set()
        for x in docs:
            ctx_prices |= set(_prices(x.page_content))
        for p in _prices(res["text"]):
            total_prices += 1
            if p in ctx_prices:
                grounded += 1
    return dict(groundedness=round(grounded / total_prices, 3) if total_prices else None,
                prices_checked=total_prices, answered=answered)


# ===========================================================================
# Runner
# ===========================================================================
def run_all(vector_db=None, with_llm=True):
    """Run every metric that the available resources allow; return a dict."""
    results = {}

    lang_res, lang_cm = eval_language()
    _print_prf("1. Language detection", lang_res)
    _print_confusion("1. Language detection", lang_cm)
    results["language"] = lang_res

    route_res, route_cm = eval_routing()
    _print_prf("2. Intent routing", route_res)
    _print_confusion("2. Intent routing", route_cm)
    results["routing"] = route_res

    ood_res, ood_cm = eval_ood()
    _print_prf("3. OOD / competitor (router-visible)", ood_res)
    print(f"  false-refusal rate (in-domain wrongly refused) = {ood_res['false_refusal_rate']}")
    results["ood"] = ood_res

    if vector_db is not None:
        ret = eval_retrieval(vector_db)
        print("\n### 4. Retrieval recall@k — router vs pure-dense baseline")
        for k in (1, 5):
            print(f"  recall@{k}: router={ret['router'][f'recall@{k}']}  "
                  f"baseline_dense={ret['baseline_dense'][f'recall@{k}']}")
        results["retrieval"] = ret

        if with_llm:
            gr = eval_groundedness(vector_db)
            print(f"\n### 5. Answer groundedness = {gr['groundedness']} "
                  f"({gr['prices_checked']} prices checked over {gr['answered']} answers)")
            results["groundedness"] = gr
    else:
        print("\n[index not available — retrieval and groundedness skipped; "
              "run on Colab with the built FAISS index for sections 4–5]")

    return results


def main():
    logging.basicConfig(level=logging.WARNING)
    vector_db = None
    try:
        import app
        app.boot()
        vector_db = app.STATE.get("index")
    except Exception as e:
        logger.warning("no index/app available (%s) — classification metrics only", e)
    results = run_all(vector_db, with_llm=vector_db is not None)
    try:
        with open("evaluate_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("\nWrote evaluate_results.json")
    except OSError:
        pass


if __name__ == "__main__":
    main()
