"""Train and predict under a leakage-free protocol.

Examples
--------
Leave-one-patient-out, every subject (one model per held-out subject)::

    python -m src.train --processed_dir data/processed_v3 --protocol lopo --out_dir results/lopo

Just a few folds for a quick look::

    python -m src.train --processed_dir data/processed_v3 --protocol lopo --folds chb01 chb05 --epochs 10

Patient-specific, forward in time (train on early seizures, test on later ones)::

    python -m src.train --processed_dir data/processed_v3 --protocol chrono --out_dir results/chrono

Each fold writes ``<out_dir>/<fold>/predictions.npz`` (validation and test risk scores for
every window, in recording order), ``model.pt`` and ``fold.json``. Then run
``python -m src.evaluate --results_dir <out_dir>``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, Sampler
from torch_geometric.data import Batch

from .config import CFG
from .data import SubjectData, list_subjects, load_subject
from .model import SoftSeizureLoss, STGNN_Soft, window_to_graph
from .splits import chronological_split, lopo_folds


# ── Dataset ──────────────────────────────────────────────────────────────────

class GraphDataset(Dataset):
    """Windows from one or more subjects, turned into PLV graphs on the fly."""

    def __init__(self, parts: list[SubjectData], norm: str = "window"):
        self.parts = parts
        self.norm = norm
        self.part_id = np.concatenate([np.full(len(p), k) for k, p in enumerate(parts)]).astype(np.int64)
        self.local = np.concatenate([np.arange(len(p)) for p in parts]).astype(np.int64)
        self.hard = np.concatenate([p.hard for p in parts]).astype(np.int64)

    def __len__(self):
        return len(self.local)

    def __getitem__(self, k):
        p = self.parts[self.part_id[k]]
        i = self.local[k]
        x, plv = p.window(i)
        x = np.asarray(x, dtype=np.float32)
        if self.norm == "window":           # per-channel z-score: removes amplitude differences
            x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)
        return window_to_graph(x, np.asarray(plv, dtype=np.float32), int(p.hard[i]), float(p.risk[i]))


class BalancedEpochSampler(Sampler):
    """Each epoch draws ``n`` windows, half preictal and half interictal, with replacement.

    Replaces the weighted sampler: the training pool can contain hundreds of thousands of
    interictal windows, so a fixed-size balanced epoch keeps epochs short and comparable.
    """

    def __init__(self, hard: np.ndarray, n: int, seed: int = 0):
        self.pos = np.where(hard == 1)[0]
        self.neg = np.where(hard == 0)[0]
        self.n = n
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __iter__(self):
        half = self.n // 2
        idx = np.concatenate([self.rng.choice(self.pos, half), self.rng.choice(self.neg, self.n - half)])
        self.rng.shuffle(idx)
        return iter(idx.tolist())


def collate(batch):
    return Batch.from_data_list(batch)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def validation_subset(part: SubjectData, max_interictal: int, seed: int) -> SubjectData:
    """All preictal windows plus a fixed random sample of interictal ones (for per-epoch AUC)."""
    neg = np.where(part.hard == 0)[0]
    if len(neg) > max_interictal:
        neg = np.sort(np.random.default_rng(seed).choice(neg, max_interictal, replace=False))
    return part.subset(np.sort(np.concatenate([np.where(part.hard == 1)[0], neg])))


# ── Train / predict ──────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, part: SubjectData, device, args, amp_dtype) -> np.ndarray:
    """Risk scores for one subject's windows, in recording order."""
    model.eval()
    loader = DataLoader(GraphDataset([part], args.norm), batch_size=args.batch_size * 2,
                        shuffle=False, collate_fn=collate, num_workers=args.num_workers)
    out = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(batch)
        out.append(torch.sigmoid(logits.float()).view(-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def train_fold(train_parts, val_parts, args, device, amp_dtype):
    seed_everything(args.seed)
    ds = GraphDataset(train_parts, args.norm)
    if (ds.hard == 1).sum() == 0:
        raise RuntimeError("training set has no preictal windows")
    loader = DataLoader(ds, batch_size=args.batch_size,
                        sampler=BalancedEpochSampler(ds.hard, args.samples_per_epoch, args.seed),
                        collate_fn=collate, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    val_small = [validation_subset(p, args.val_max_interictal, args.seed) for p in val_parts]

    model = STGNN_Soft().to(device)
    crit = SoftSeizureLoss(args.alpha, args.pos_weight, args.mse_on_logits).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_auc, best_state, history, stale = -1.0, None, [], 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(batch)
            loss = crit(logits.float(), batch.risk, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            opt.step()
            total += loss.item()
        sched.step()

        # Model selection on validation subjects only (never the test subject)
        probs = np.concatenate([predict(model, p, device, args, amp_dtype) for p in val_small])
        hard = np.concatenate([p.hard for p in val_small])
        val_auc = roc_auc_score(hard, probs) if len(np.unique(hard)) > 1 else float("nan")
        history.append({"epoch": epoch, "train_loss": total / max(1, len(loader)), "val_auc": val_auc})
        flag = ""
        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc, stale = val_auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = "  *"
        else:
            stale += 1
        print(f"    ep {epoch:02d}  loss {history[-1]['train_loss']:.4f}  "
              f"val AUC {val_auc:.4f}  [{time.time() - t0:.0f}s]{flag}", flush=True)
        if args.patience and stale >= args.patience:
            print(f"    early stop (no val improvement for {args.patience} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_auc


def save_fold(out_dir, name, model, history, best_auc, val_parts, test_parts, args, device, amp_dtype):
    fold_dir = os.path.join(out_dir, name)
    os.makedirs(fold_dir, exist_ok=True)
    arrays = {}
    for split, parts in (("val", val_parts), ("test", test_parts)):
        for p in parts:
            key = f"{split}__{p.subject}"
            arrays[f"{key}__probs"] = predict(model, p, device, args, amp_dtype)
            arrays[f"{key}__hard"] = p.hard
            arrays[f"{key}__block"] = p.block
            arrays[f"{key}__run"] = p.run
            if p.t_start is not None:
                arrays[f"{key}__t"] = p.t_start
    np.savez_compressed(os.path.join(fold_dir, "predictions.npz"), **arrays)
    torch.save({"model_state": model.state_dict(), "val_auc": best_auc}, os.path.join(fold_dir, "model.pt"))
    with open(os.path.join(fold_dir, "fold.json"), "w") as f:
        json.dump({"fold": name, "protocol": args.protocol,
                   "val": [p.subject for p in val_parts], "test": [p.subject for p in test_parts],
                   "best_val_auc": best_auc, "history": history, "args": vars(args)},
                  f, indent=2, default=str)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--protocol", choices=["lopo", "chrono"], default="lopo")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--folds", nargs="*", default=None, help="subjects to run (default: all)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8, help="early stopping on val AUC (0 = off)")
    ap.add_argument("--samples_per_epoch", type=int, default=40000)
    ap.add_argument("--batch_size", type=int, default=CFG["batch_size"])
    ap.add_argument("--lr", type=float, default=CFG["lr"])
    ap.add_argument("--alpha", type=float, default=CFG["alpha"])
    ap.add_argument("--pos_weight", type=float, default=1.0,
                    help="BCE positive weight (1.0: epochs are already class-balanced)")
    ap.add_argument("--mse_on_logits", action="store_true",
                    help="reproduce the v2 loss exactly (MSE on raw logits)")
    ap.add_argument("--norm", choices=["window", "none"], default="window")
    ap.add_argument("--n_val", type=int, default=3, help="validation subjects per LOPO fold")
    ap.add_argument("--val_max_interictal", type=int, default=20000)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=CFG["seed"])
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args(argv)
    args.out_dir = args.out_dir or os.path.join("results", args.protocol)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and not args.no_amp else None
    print(f"device={device}  amp={amp_dtype}  protocol={args.protocol}")

    subjects = list_subjects(args.processed_dir)
    data = {s: load_subject(args.processed_dir, s) for s in subjects}
    w = CFG["window_sec"]
    for s, d in data.items():
        print(f"  {s}: {len(d):7d} windows  seizures={d.n_seizures:2d}  "
              f"interictal={np.sum(d.hard == 0) * w / 3600:6.1f} h  "
              f"preictal={np.sum(d.hard == 1) * w / 3600:4.1f} h")

    if args.protocol == "lopo":
        folds = lopo_folds(subjects, {s: d.n_seizures for s, d in data.items()},
                           n_val=args.n_val, seed=args.seed)
        if args.folds:
            folds = [f for f in folds if f.name in args.folds]
        for f in folds:
            print(f"\n== fold {f.name}: train {len(f.train)} subj | val {f.val} | test {f.test}")
            tr, va, te = ([data[s] for s in grp] for grp in (f.train, f.val, f.test))
            model, hist, auc = train_fold(tr, va, args, device, amp_dtype)
            save_fold(args.out_dir, f.name, model, hist, auc, va, te, args, device, amp_dtype)
    else:
        for s in args.folds or subjects:
            d = data[s]
            sp = chronological_split(d.block)
            if sp is None:
                print(f"\n== {s}: skipped ({d.n_seizures} seizures; need >= 3)")
                continue
            print(f"\n== {s}: train {len(sp['train'])} | val {len(sp['val'])} | test {len(sp['test'])} windows")
            tr, va, te = [d.subset(sp["train"])], [d.subset(sp["val"])], [d.subset(sp["test"])]
            model, hist, auc = train_fold(tr, va, args, device, amp_dtype)
            save_fold(args.out_dir, s, model, hist, auc, va, te, args, device, amp_dtype)

    print(f"\nDone. Now run:  python -m src.evaluate --results_dir {args.out_dir}")


if __name__ == "__main__":
    main()
