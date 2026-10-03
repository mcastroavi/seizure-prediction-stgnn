"""Non-deep baseline: logistic regression on PLV and channel-power features.

    python -m src.baseline --processed_dir data/processed_v3 --protocol lopo --out_dir results/baseline_lopo
    python -m src.evaluate --results_dir results/baseline_lopo

Uses exactly the same folds, validation subjects and output format as ``src.train``, so
``src.evaluate`` and ``src.figures`` work unchanged. Answering "does the graph network beat
a simple model on the same features?" is part of an honest evaluation. Runs on CPU in minutes.

Features per window: the 153 upper-triangle PLV values (18 channels) plus the 18
per-channel log variances (a broadband power proxy). Standardised on training data only.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .data import SubjectData, list_subjects, load_subject
from .splits import chronological_split, lopo_folds


def features(part: SubjectData, chunk: int = 4096) -> np.ndarray:
    n_ch = part.plv.shape[1]
    iu = np.triu_indices(n_ch, 1)
    out = []
    for s in range(0, len(part), chunk):
        sel = part.sel[s:s + chunk]
        order = np.argsort(sel)                      # sorted reads are much faster on memmaps
        plv = np.asarray(part.plv[sel[order]], dtype=np.float32)[np.argsort(order)]
        x = np.asarray(part.X[sel[order]], dtype=np.float32)[np.argsort(order)]
        out.append(np.hstack([plv[:, iu[0], iu[1]], np.log(x.var(axis=2) + 1e-6)]))
    return np.vstack(out) if out else np.zeros((0, len(iu[0]) + n_ch), np.float32)


def sample_train(parts, max_interictal_per_subject, seed):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for p in parts:
        pos = np.where(p.hard == 1)[0]
        neg = np.where(p.hard == 0)[0]
        if len(neg) > max_interictal_per_subject:
            neg = rng.choice(neg, max_interictal_per_subject, replace=False)
        idx = np.sort(np.concatenate([pos, neg]))
        sub = p.subset(idx)
        Xs.append(features(sub)); ys.append(sub.hard)
    return np.vstack(Xs), np.concatenate(ys)


def run_fold(name, train_parts, val_parts, test_parts, args, protocol):
    Xtr, ytr = sample_train(train_parts, args.max_interictal, args.seed)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=args.C, class_weight="balanced", max_iter=2000)
    clf.fit(scaler.transform(Xtr), ytr)

    arrays, val_auc = {}, []
    for split, parts in (("val", val_parts), ("test", test_parts)):
        for p in parts:
            probs = clf.predict_proba(scaler.transform(features(p)))[:, 1].astype(np.float32)
            key = f"{split}__{p.subject}"
            arrays.update({f"{key}__probs": probs, f"{key}__hard": p.hard,
                           f"{key}__block": p.block, f"{key}__run": p.run})
            if p.t_start is not None:
                arrays[f"{key}__t"] = p.t_start
            if split == "val" and len(np.unique(p.hard)) > 1:
                val_auc.append(roc_auc_score(p.hard, probs))
    d = os.path.join(args.out_dir, name)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, "predictions.npz"), **arrays)
    with open(os.path.join(d, "fold.json"), "w") as f:
        json.dump({"fold": name, "protocol": protocol, "model": "logistic_regression",
                   "val": [p.subject for p in val_parts], "test": [p.subject for p in test_parts],
                   "val_auc_mean": float(np.mean(val_auc)) if val_auc else None,
                   "args": vars(args)}, f, indent=2)
    print(f"  {name}: val AUC {np.mean(val_auc) if val_auc else float('nan'):.3f}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--protocol", choices=["lopo", "chrono"], default="lopo")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--n_val", type=int, default=3)
    ap.add_argument("--max_interictal", type=int, default=20000, help="per training subject")
    ap.add_argument("--C", type=float, default=0.1, help="inverse L2 regularisation strength")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    args.out_dir = args.out_dir or os.path.join("results", f"baseline_{args.protocol}")

    subjects = list_subjects(args.processed_dir)
    data = {s: load_subject(args.processed_dir, s) for s in subjects}
    if args.protocol == "lopo":
        folds = lopo_folds(subjects, {s: d.n_seizures for s, d in data.items()}, args.n_val, args.seed)
        for f in folds:
            if args.folds and f.name not in args.folds:
                continue
            run_fold(f.name, [data[s] for s in f.train], [data[s] for s in f.val],
                     [data[s] for s in f.test], args, "lopo")
    else:
        for s in args.folds or subjects:
            sp = chronological_split(data[s].block)
            if sp is None:
                continue
            d = data[s]
            run_fold(s, [d.subset(sp["train"])], [d.subset(sp["val"])], [d.subset(sp["test"])],
                     args, "chrono")
    print(f"Done. Now run:  python -m src.evaluate --results_dir {args.out_dir}")


if __name__ == "__main__":
    main()
