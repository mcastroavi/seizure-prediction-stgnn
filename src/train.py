"""Train and predict under a leakage-free protocol.

Examples
--------
Leave-one-patient-out, every subject (one model per held-out subject)::

    python -m src.train --processed_dir data/processed --protocol lopo --out_dir results/lopo

Just a few folds for a quick look::

    python -m src.train --processed_dir data/processed --protocol lopo --folds chb01 chb05 --epochs 20

Patient-specific, forward in time (train on early seizures, test on later ones)::

    python -m src.train --processed_dir data/processed --protocol chrono --out_dir results/chrono

Each fold writes ``<out_dir>/<fold>/predictions.npz`` (validation and test risk scores in
recording order) and ``model.pt``. Then run ``python -m src.evaluate --results_dir <out_dir>``.
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch_geometric.data import Batch

from .config import CFG
from .data import SubjectData, list_subjects, load_subject
from .model import SoftSeizureLoss, STGNN_Soft, window_to_graph
from .splits import chronological_split, lopo_folds


# ── Dataset ──────────────────────────────────────────────────────────────────

class GraphDataset(Dataset):
    def __init__(self, parts: list[SubjectData]):
        self.parts = parts
        self.index = [(p, i) for p, part in enumerate(parts) for i in range(len(part))]
        self.hard = np.concatenate([p.hard for p in parts]) if parts else np.zeros(0, int)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        p, i = self.index[k]
        s = self.parts[p]
        return window_to_graph(s.X[i], s.plv[i], int(s.hard[i]), float(s.risk[i]))


def collate(batch):
    return Batch.from_data_list(batch)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Train / predict ──────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, part: SubjectData, device, batch_size, amp_dtype) -> np.ndarray:
    """Risk scores for one subject's windows, in recording order."""
    model.eval()
    loader = DataLoader(GraphDataset([part]), batch_size=batch_size, shuffle=False,
                        collate_fn=collate)
    out = []
    for batch in loader:
        batch = batch.to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(batch)
        out.append(torch.sigmoid(logits.float()).view(-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def train_fold(train_parts, val_parts, args, device, amp_dtype):
    seed_everything(args.seed)
    ds = GraphDataset(train_parts)
    counts = np.bincount(ds.hard, minlength=2).astype(float)
    weights = 1.0 / counts[ds.hard]
    sampler = WeightedRandomSampler(torch.as_tensor(weights), num_samples=len(weights),
                                    replacement=True)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                        collate_fn=collate, num_workers=0,
                        pin_memory=device.type == "cuda")

    model = STGNN_Soft().to(device)
    crit = SoftSeizureLoss(args.alpha, args.pos_weight, args.mse_on_logits).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_auc, best_state, history = -1.0, None, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(batch)
            loss = crit(logits.float(), batch.risk, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            opt.step()
            total += loss.item()
        sched.step()

        # Model selection on validation subjects only (never the test subject)
        probs = np.concatenate([predict(model, p, device, args.batch_size, amp_dtype) for p in val_parts])
        hard = np.concatenate([p.hard for p in val_parts])
        val_auc = roc_auc_score(hard, probs) if len(np.unique(hard)) > 1 else float("nan")
        history.append({"epoch": epoch, "train_loss": total / max(1, len(loader)), "val_auc": val_auc})
        flag = ""
        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = "  *"
        print(f"    ep {epoch:02d}  loss {history[-1]['train_loss']:.4f}  "
              f"val AUC {val_auc:.4f}  [{time.time() - t0:.0f}s]{flag}", flush=True)

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
            arrays[f"{key}__probs"] = predict(model, p, device, args.batch_size, amp_dtype)
            arrays[f"{key}__hard"] = p.hard
            arrays[f"{key}__block"] = p.block
            arrays[f"{key}__run"] = p.run
    np.savez_compressed(os.path.join(fold_dir, "predictions.npz"), **arrays)
    torch.save({"model_state": model.state_dict(), "val_auc": best_auc}, os.path.join(fold_dir, "model.pt"))
    with open(os.path.join(fold_dir, "fold.json"), "w") as f:
        json.dump({"fold": name, "protocol": args.protocol,
                   "val": [p.subject for p in val_parts],
                   "test": [p.subject for p in test_parts],
                   "best_val_auc": best_auc, "history": history,
                   "args": vars(args)}, f, indent=2, default=str)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--protocol", choices=["lopo", "chrono"], default="lopo")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--folds", nargs="*", default=None, help="subjects to run (default: all)")
    ap.add_argument("--epochs", type=int, default=CFG["epochs"])
    ap.add_argument("--batch_size", type=int, default=CFG["batch_size"])
    ap.add_argument("--lr", type=float, default=CFG["lr"])
    ap.add_argument("--alpha", type=float, default=CFG["alpha"])
    ap.add_argument("--pos_weight", type=float, default=CFG["pos_weight"])
    ap.add_argument("--mse_on_logits", action="store_true",
                    help="reproduce the notebook's loss exactly (MSE on raw logits)")
    ap.add_argument("--n_val", type=int, default=3, help="validation subjects per LOPO fold")
    ap.add_argument("--seed", type=int, default=CFG["seed"])
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args(argv)
    args.out_dir = args.out_dir or os.path.join("results", args.protocol)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and not args.no_amp else None
    print(f"device={device}  amp={amp_dtype}  protocol={args.protocol}")

    subjects = list_subjects(args.processed_dir)
    print(f"Loading {len(subjects)} subjects ...", flush=True)
    data = {s: load_subject(args.processed_dir, s) for s in subjects}
    for s, d in data.items():
        print(f"  {s}: {len(d):6d} windows  seizures={d.n_seizures:2d}  "
              f"interictal={np.sum(d.hard == 0) * CFG['window_sec'] / 3600:.2f} h")

    if args.protocol == "lopo":
        folds = lopo_folds(subjects, {s: d.n_seizures for s, d in data.items()},
                           n_val=args.n_val, seed=args.seed)
        if args.folds:
            folds = [f for f in folds if f.name in args.folds]
        for f in folds:
            print(f"\n== fold {f.name}: train {len(f.train)} subj | val {f.val} | test {f.test}")
            train_parts = [data[s] for s in f.train]
            val_parts = [data[s] for s in f.val]
            test_parts = [data[s] for s in f.test]
            model, hist, auc = train_fold(train_parts, val_parts, args, device, amp_dtype)
            save_fold(args.out_dir, f.name, model, hist, auc, val_parts, test_parts, args, device, amp_dtype)

    else:  # chrono
        names = args.folds or subjects
        for s in names:
            sp = chronological_split(data[s].block)
            if sp is None:
                print(f"\n== {s}: skipped ({data[s].n_seizures} seizures; need >= 3)")
                continue
            d = data[s]
            print(f"\n== {s}: train {len(sp['train'])} | val {len(sp['val'])} | test {len(sp['test'])} windows")
            train_parts = [d.subset(sp["train"])]
            val_parts = [d.subset(sp["val"])]
            test_parts = [d.subset(sp["test"])]
            model, hist, auc = train_fold(train_parts, val_parts, args, device, amp_dtype)
            save_fold(args.out_dir, s, model, hist, auc, val_parts, test_parts, args, device, amp_dtype)

    print(f"\nDone. Now run:  python -m src.evaluate --results_dir {args.out_dir}")


if __name__ == "__main__":
    main()
