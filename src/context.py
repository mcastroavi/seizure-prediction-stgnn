"""Context model: predict from the last few minutes of windows, not one 5-second window.

    # on top of a trained ST-GNN (uses each fold's model.pt to embed windows)
    python -m src.context --processed_dir data/processed_v3 --source results/stgnn_bands_chrono \\
        --out_dir results/context_stgnn_bands_chrono
    # on hand-crafted band features (no deep encoder)
    python -m src.context --processed_dir data/processed_v3 --protocol chrono --features bands \\
        --out_dir results/context_feat_chrono

Each window is first turned into a vector: the 64-d embedding of that fold's trained ST-GNN
(``--source``) or standardised hand-crafted features (broadband + band PLV, band power).
A small GRU then reads the last ``--seq_len`` windows (default 60 = 5 minutes) and outputs
the risk for the current window. Sequences never cross a recording gap (shorter histories
are zero-padded, with a mask channel), and only past windows are used, so the model is causal.

Folds, validation subjects and output format match ``src.train``: ``src.evaluate`` and
``src.figures`` work unchanged.

Caveat (stacking): with ``--source``, training windows are embedded by an encoder that was
trained on them, so their embeddings are cleaner than those of unseen windows. Model selection
on held-out validation data guards against relying on that.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from . import baseline as bl
from .context_features import extra_features
from .data import SubjectData, list_subjects, load_subject
from .model import SoftSeizureLoss, STGNN_Soft
from .splits import chronological_split, lopo_folds
from .train import collate, make_dataset


# ── Window vectors ───────────────────────────────────────────────────────────

@torch.no_grad()
def embed_part(model, part: SubjectData, enc_args, device, amp_dtype, batch_size=512) -> np.ndarray:
    model.eval()
    loader = DataLoader(make_dataset([part], enc_args), batch_size=batch_size, shuffle=False,
                        collate_fn=collate, num_workers=enc_args.num_workers)
    out = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out.append(model.embed(batch).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 64), np.float32)


def load_encoder(path, device):
    ck = torch.load(path, map_location=device)
    model = STGNN_Soft(**ck.get("model_kwargs", {})).to(device)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model_state"].items()})
    enc_args = SimpleNamespace(norm=ck.get("norm", "window"), features=ck.get("features", "v3"),
                               full_graph=ck.get("full_graph", True), num_workers=4)
    return model, enc_args


# ── Sequences ────────────────────────────────────────────────────────────────

class Stack:
    """Window vectors of several parts concatenated, with per-window run starts so that
    histories never cross a recording gap or a part boundary."""

    def __init__(self, parts: list[SubjectData], vecs: list[np.ndarray], device):
        self.parts = parts
        self.E = torch.as_tensor(np.concatenate(vecs), dtype=torch.float16, device=device)
        starts, off = [], 0
        for p in parts:
            r = p.run
            b = np.concatenate([[0], np.where(np.diff(r) != 0)[0] + 1])
            idx = np.arange(len(r))
            starts.append(b[np.searchsorted(b, idx, side="right") - 1] + off)
            off += len(r)
        self.run_start = torch.as_tensor(np.concatenate(starts), device=device)
        self.hard = torch.as_tensor(np.concatenate([p.hard for p in parts]), device=device)
        self.risk = torch.as_tensor(np.concatenate([p.risk for p in parts]), dtype=torch.float32, device=device)
        self.bounds = np.cumsum([0] + [len(p) for p in parts])

    def batch(self, t: torch.Tensor, L: int) -> torch.Tensor:
        """(B,) anchor indices -> (B, L, D + 1) zero-padded histories + validity mask."""
        offs = torch.arange(L - 1, -1, -1, device=t.device)
        idx = t[:, None] - offs[None, :]
        valid = idx >= self.run_start[t][:, None]
        x = self.E[idx.clamp(min=0)].float() * valid[..., None]
        return torch.cat([x, valid[..., None].float()], dim=-1)


class ContextGRU(nn.Module):
    def __init__(self, d_in, hidden=64, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in + 1, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        h, _ = self.gru(self.proj(x))
        return self.head(h[:, -1])


@torch.no_grad()
def predict_stack(model, stack: Stack, L, bs=4096):
    model.eval()
    out = []
    for s in range(0, len(stack.hard), bs):
        t = torch.arange(s, min(s + bs, len(stack.hard)), device=stack.E.device)
        out.append(torch.sigmoid(model(stack.batch(t, L)).float()).view(-1).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, np.float32)


def train_context(tr: Stack, va: Stack, args, device, init_state=None):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = ContextGRU(tr.E.shape[1], args.hidden, args.dropout).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    crit = SoftSeizureLoss(alpha=0.5).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    hard = tr.hard.cpu().numpy()
    pos, neg = np.where(hard == 1)[0], np.where(hard == 0)[0]

    # validation: all preictal + a fixed sample of interictal windows
    vh = va.hard.cpu().numpy()
    vneg = np.where(vh == 0)[0]
    if len(vneg) > 20000:
        vneg = rng.choice(vneg, 20000, replace=False)
    vidx = torch.as_tensor(np.sort(np.concatenate([np.where(vh == 1)[0], vneg])), device=device)

    best, best_state, stale, hist = -1.0, None, 0, []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        anchors = np.concatenate([rng.choice(pos, args.samples_per_epoch // 2),
                                  rng.choice(neg, args.samples_per_epoch // 2)])
        rng.shuffle(anchors)
        anchors = torch.as_tensor(anchors, device=device)
        tot = 0.0
        for s in range(0, len(anchors), args.batch_size):
            t = anchors[s:s + args.batch_size]
            loss = crit(model(tr.batch(t, args.seq_len)), tr.risk[t], tr.hard[t])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(va.batch(vidx, args.seq_len)).float()).view(-1).cpu().numpy()
        y = vh[vidx.cpu().numpy()]
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
        hist.append({"epoch": ep, "loss": tot, "val_auc": auc})
        flag = ""
        if not np.isnan(auc) and auc > best:
            best, stale, flag = auc, 0, "  *"
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        print(f"    ep {ep:02d}  val AUC {auc:.4f}  [{time.time() - t0:.1f}s]{flag}", flush=True)
        if stale >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best


# ── Folds ────────────────────────────────────────────────────────────────────

def fold_parts(name, fold_info, data, protocol):
    if protocol == "chrono":
        d = data[name]
        sp = chronological_split(d.block)
        return [d.subset(sp["train"])], [d.subset(sp["val"])], [d.subset(sp["test"])]
    return ([data[s] for s in fold_info["train"]], [data[s] for s in fold_info["val"]],
            [data[s] for s in fold_info["test"]])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--processed_dir", required=True)
    ap.add_argument("--source", default=None, help="results dir of src.train (window encoder per fold)")
    ap.add_argument("--protocol", choices=["lopo", "chrono"], default=None,
                    help="required without --source; with --source it is read from the folds")
    ap.add_argument("--features", choices=["v3", "bands"], default="bands",
                    help="hand-crafted features when no --source is given")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--folds", nargs="*", default=None)
    ap.add_argument("--seq_len", type=int, default=60, help="windows of history (60 x 5 s = 5 min)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--samples_per_epoch", type=int, default=40000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_val", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extra", default="",
                    help="comma-separated extra inputs per window: time (hour of day), "
                         "history (time since last seizure, floored at the labelling buffer)")
    args = ap.parse_args(argv)
    extras = [x for x in args.extra.split(",") if x]
    for x in extras:
        if x not in ("time", "history"):
            ap.error(f"unknown --extra '{x}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.bfloat16 if device.type == "cuda" else None
    subjects = list_subjects(args.processed_dir)
    data = {s: load_subject(args.processed_dir, s) for s in subjects}

    # folds: from the encoder's results dir, or recomputed exactly as src.train does
    folds = {}
    if args.source:
        for f in sorted(os.listdir(args.source)):
            fj = os.path.join(args.source, f, "fold.json")
            if os.path.exists(fj):
                folds[f] = json.load(open(fj))
        protocol = next(iter(folds.values()))["protocol"]
        for name, info in folds.items():
            if protocol == "lopo" and not info.get("train"):
                info["train"] = [s for s in subjects if s not in info["val"] + info["test"]]
    else:
        if not args.protocol:
            ap.error("--protocol is required without --source")
        protocol = args.protocol
        if protocol == "lopo":
            for f in lopo_folds(subjects, {s: d.n_seizures for s, d in data.items()}, args.n_val, args.seed):
                folds[f.name] = {"train": f.train, "val": f.val, "test": f.test}
        else:
            folds = {s: {} for s in subjects if chronological_split(data[s].block) is not None}
    if args.folds:
        folds = {k: v for k, v in folds.items() if k in args.folds}

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"context model | protocol={protocol} | vectors="
          f"{'ST-GNN embeddings from ' + args.source if args.source else 'hand-crafted ' + args.features} "
          f"| history {args.seq_len} windows ({args.seq_len * 5 / 60:.0f} min)"
          f"{' | extra: ' + ','.join(extras) if extras else ''}")

    for name, info in folds.items():
        t0 = time.time()
        tr, va, te = fold_parts(name, info, data, protocol)
        if args.source:
            enc, enc_args = load_encoder(os.path.join(args.source, name, "model.pt"), device)
            vec = {id(p): embed_part(enc, p, enc_args, device, amp) for p in tr + va + te}
            del enc
        else:
            raw = {id(p): bl.features(p, kind=args.features) for p in tr + va + te}
            fit = np.concatenate([raw[id(p)] for p in tr])
            if len(fit) > 200000:
                fit = fit[np.random.default_rng(args.seed).choice(len(fit), 200000, replace=False)]
            scaler = StandardScaler().fit(fit)
            vec = {k: np.clip(scaler.transform(v), -8, 8).astype(np.float32) for k, v in raw.items()}
        if extras:
            vec = {id(p): np.concatenate([vec[id(p)], extra_features(p, extras)], axis=1).astype(np.float32)
                   for p in tr + va + te}
        print(f"\n== {name}: vectors ready in {time.time() - t0:.0f}s "
              f"(train {sum(len(p) for p in tr)} | val {sum(len(p) for p in va)} | test {sum(len(p) for p in te)})")

        S_tr = Stack(tr, [vec[id(p)] for p in tr], device)
        S_va = Stack(va, [vec[id(p)] for p in va], device)
        model, hist, best = train_context(S_tr, S_va, args, device)

        arrays = {}
        for split, parts in (("val", va), ("test", te)):
            for p in parts:
                probs = predict_stack(model, Stack([p], [vec[id(p)]], device), args.seq_len)
                key = f"{split}__{p.subject}"
                arrays.update({f"{key}__probs": probs, f"{key}__hard": p.hard,
                               f"{key}__block": p.block, f"{key}__run": p.run})
                if p.t_start is not None:
                    arrays[f"{key}__t"] = p.t_start
        fd = os.path.join(args.out_dir, name)
        os.makedirs(fd, exist_ok=True)
        np.savez_compressed(os.path.join(fd, "predictions.npz"), **arrays)
        torch.save({"model_state": model.state_dict(), "d_in": S_tr.E.shape[1], "args": vars(args)},
                   os.path.join(fd, "context.pt"))
        with open(os.path.join(fd, "fold.json"), "w") as fh:
            json.dump({"fold": name, "protocol": protocol, "model": "context_gru",
                       "vectors": args.source or f"handcrafted_{args.features}", "extra": extras,
                       "train": [p.subject for p in tr] if protocol == "lopo" else [name],
                       "val": [p.subject for p in va], "test": [p.subject for p in te],
                       "best_val_auc": best, "history": hist, "args": vars(args)}, fh, indent=2)
        del S_tr, S_va
        torch.cuda.empty_cache() if device.type == "cuda" else None

    print(f"\nDone. Now run:  python -m src.evaluate --results_dir {args.out_dir}")


if __name__ == "__main__":
    main()
