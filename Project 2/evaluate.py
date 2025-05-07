import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve
import os

# — (your compute_average_precision, compute_mean_average_precision, plot_precision_recall_curve go here) —

def evaluate_keyword_spotting(keyword_index, validation_words, all_results, output_dir=None):
    """
    Evaluate keyword spotting performance.
    Args:
        keyword_index (dict): mapping keyword → list of all word IDs with that transcription
        validation_words (set): set of word IDs in the validation split
        all_results (dict): mapping query word ID → list of (word_id, dtw_score) sorted ascending
        output_dir (str, optional): where to save metrics & plots
    Returns:
        dict: { 'mAP':…, 'per_query_AP':…, 'per_keyword_AP':… }
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # build relevancy mapping: for each query q, relevant = same-keyword in validation minus itself
    all_relevant = {}
    for kw, instances in keyword_index.items():
        val_insts = [w for w in instances if w in validation_words]
        for q in instances:
            if q in validation_words:
                all_relevant[q] = [w for w in val_insts if w != q]

    # compute per‑query AP
    per_query_AP = {}
    for q, results in all_results.items():
        rel = all_relevant.get(q, [])
        per_query_AP[q] = compute_average_precision(results, rel)

    # compute mAP
    mAP = np.mean(list(per_query_AP.values())) if per_query_AP else 0.0

    # compute per‑keyword AP
    per_keyword_AP = {}
    for kw, instances in keyword_index.items():
        # only those queries in validation
        qlist = [q for q in instances if q in per_query_AP]
        if qlist:
            per_keyword_AP[kw] = np.mean([per_query_AP[q] for q in qlist])

    # save numeric results
    if output_dir:
        with open(os.path.join(output_dir, "evaluation_results.txt"), "w") as f:
            f.write(f"mAP: {mAP:.4f}\n\nPer-keyword AP:\n")
            for kw, ap in sorted(per_keyword_AP.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{kw}\t{ap:.4f}\n")

    # plot PR curve for each keyword (first N keywords or all)
    for kw, ap in per_keyword_AP.items():
        # pick one representative query (first)
        q = next(q for q in keyword_index[kw] if q in all_results)
        prec, rec = plot_precision_recall_curve(all_results[q],
                                                all_relevant.get(q, []),
                                                output_path=(os.path.join(output_dir, f"PR_{kw}.png") if output_dir else None))
    return {
        'mAP': mAP,
        'per_query_AP': per_query_AP,
        'per_keyword_AP': per_keyword_AP
    }

# ── DRIVER ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dtw import dtw_distance
    from features import extract_features

    # 1) load transcription and splits
    import csv
    # load transcription
    id2txt = {}
    with open("documents/transcription.tsv") as f:
        for wid, seq in csv.reader(f, delimiter='\t'):
            id2txt[wid] = seq.replace('-', '')
    # load train/validation split
    val_set = set()
    with open("documents/validation.tsv") as f:
        for row in f:
            val_set.add(row.strip())

    # 2) build keyword index from keywords.tsv
    keyword_index = {}
    with open("documents/keywords.tsv") as f:
        for kw in f:
            keyword_index[kw.strip()] = [wid for wid,t in id2txt.items() if t==kw.strip()]

    # 3) prepare all_results: for each validation word as query, rank all word_images
    all_results = {}
    # pre‑extract all features
    feats = {}
    for wid in val_set.union(*keyword_index.values()):
        path = f"documents/word_images/{wid}.png"
        if os.path.exists(path):
            feats[wid] = extract_features(path)

    for q in sorted(val_set):
        if q not in feats: continue
        qf = feats[q]
        dlist = []
        for tid, tf in feats.items():
            dlist.append((tid, dtw_distance(qf, tf)))
        all_results[q] = sorted(dlist, key=lambda x: x[1])

    # 4) run evaluation
    metrics = evaluate_keyword_spotting(keyword_index, val_set, all_results, output_dir="eval_outputs")

    print(f"Done. mAP = {metrics['mAP']:.4f}. Detailed results in ./eval_outputs/")
