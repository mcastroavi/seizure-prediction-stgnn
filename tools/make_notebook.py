"""Build notebooks/stgnn_v3_walkthrough.ipynb (run: python tools/make_notebook.py)."""

import json
import os

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md(r"""
# ST-GNN Seizure Prediction — v3 walkthrough

This notebook runs the whole v3 pipeline one step at a time, so each stage can be inspected
and explained:

1. **Raw data** — EDF files, summary files, montage, recording timeline
2. **Preprocessing** — filtering, windows, labels, PLV graphs
3. **What the model sees** — PLV before a seizure vs. normal EEG, soft labels
4. **Model** — graph construction and the ST-GNN
5. **Leakage-free splits** — leave-one-patient-out and chronological
6. **Training one fold** — ST-GNN and the logistic-regression baseline
7. **Seizure-level evaluation** — alarms, sensitivity, false alarms/hour, chance test, figures
8. **Full results** — all four runs side by side
9. **Optional: why v2 scored 0.88** — the same model with v2's random window split

The heavy lifting lives in `src/` (tested code); each cell calls it and shows the result.
Run the cells in order. Section 2 can be skipped if `data/processed_v3` already exists.
""")

md("## 0. Setup")
code(r"""
import os, sys, json, time
from pathlib import Path
from types import SimpleNamespace

# Run from the repository root no matter where the notebook was opened
ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import torch
from IPython.display import Image, Markdown, display

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = torch.bfloat16 if DEVICE.type == "cuda" else None
print("repo   :", ROOT)
print("torch  :", torch.__version__, "| device:", DEVICE,
      f"({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else "")

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "axes.axisbelow": True})
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#52514e"
""")

md(r"""
### Configuration
Edit the paths if your data lives elsewhere. `EPOCHS` controls the demo training in
Section 6 (the full runs in `run_all.sh` use 30 with early stopping).
""")
code(r"""
RAW_DIR  = Path(os.environ.get("NB_RAW_DIR",
               "~/seizure_project/data/raw/chb-mit-scalp-eeg-database-1.0.0")).expanduser()
PROC_DIR = Path(os.environ.get("NB_PROC_DIR", "data/processed_v3"))
SUBJECT  = os.environ.get("NB_SUBJECT", "chb01")      # subject used for the walkthrough

RUN_PREPROCESS = os.environ.get("NB_RUN_PREPROCESS", "0") == "1"  # True: rebuild SUBJECT from EDF
EPOCHS         = int(os.environ.get("NB_EPOCHS", "10"))
RUN_LEAKY_DEMO = os.environ.get("NB_RUN_LEAKY", "1") == "1"       # Section 9 (~10-20 min on GPU)

print("raw data :", RAW_DIR, "| exists:", RAW_DIR.exists())
print("processed:", PROC_DIR, "| exists:", PROC_DIR.exists())
""")

# ── 1. Raw data ──────────────────────────────────────────────────────────────
md(r"""
## 1. Raw data

CHB-MIT gives, per subject, ~1-hour EDF recordings plus a `chbXX-summary.txt` listing each
file's clock start time and the seizure times **relative to that file**.
""")
code(r"""
from src.chbmit import parse_summary

summary = parse_summary((RAW_DIR / SUBJECT / f"{SUBJECT}-summary.txt").read_text())
hms = lambda s: "—" if s is None else f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"
print(f"{SUBJECT}: {len(summary)} files in summary, "
      f"{sum(len(e.seizures) for e in summary.values())} seizures\n")
print(f"{'file':16s} {'start':>9s}  seizures (s from file start)")
for name, e in summary.items():
    if e.seizures:
        print(f"{name:16s} {hms(e.start_tod):>9s}  {e.seizures}")
""")

md(r"""
### Reading an EDF file and choosing channels
`src/edf.py` reads EDF in plain numpy. The pipeline keeps the **18 standard bipolar
channels**; `channel_plan` picks them by name (and can derive a bipolar channel from a
referential recording, e.g. F7-T7 = (F7-CS2) − (T7-CS2)).
""")
code(r"""
from src.edf import read_header, read_signals
from src.chbmit import channel_plan, apply_plan, STANDARD_CHANNELS

sz_file = next(n for n, e in summary.items() if e.seizures)
path = RAW_DIR / SUBJECT / sz_file
h = read_header(str(path))
print(f"{sz_file}: {len(h.labels)} signals, {h.fs[0]:.0f} Hz, {h.duration_sec/3600:.2f} h, "
      f"header start {h.start_time}")
print("header channels:", h.labels)

plan = channel_plan(h.labels)
idx = sorted({i for step in plan for i in step[1:]})
pos = {i: j for j, i in enumerate(idx)}
sig = apply_plan(read_signals(str(path), h, idx), [(s[0],) + tuple(pos[i] for i in s[1:]) for s in plan])
print("selected:", sig.shape, "(18 channels × samples)")

onset = summary[sz_file].seizures[0][0]
fs = int(h.fs[0]); t0 = int((onset - 5) * fs); t1 = int((onset + 10) * fs)
t = np.arange(t0, t1) / fs - onset
fig, ax = plt.subplots(figsize=(11, 4))
for k, ch in enumerate([0, 4, 8, 12]):
    ax.plot(t, sig[ch, t0:t1] / 300 - 2 * k, lw=0.6, color=BLUE)
    ax.text(t[0], -2 * k + 0.6, STANDARD_CHANNELS[ch], fontsize=8, color=GRAY)
ax.axvline(0, color=ORANGE, ls="--", lw=1.2); ax.text(0.2, 1.2, "seizure onset", color=ORANGE)
ax.set_yticks([]); ax.set_xlabel("seconds from onset"); ax.set_title(f"{sz_file}: raw EEG around a seizure", loc="left")
plt.show()
""")

md(r"""
### Building one continuous timeline
Seizures near a file boundary have their 30-minute preictal period in the *previous* file,
so every file gets an **absolute** start time. Clock times wrap at midnight; when a file
would start before the previous one ended, a day is added.
""")
code(r"""
from src.chbmit import build_timeline
import glob

names = sorted(os.path.basename(p) for p in glob.glob(str(RAW_DIR / SUBJECT / "*.edf")))
heads = [read_header(str(RAW_DIR / SUBJECT / n)) for n in names]
tods = [summary[n].start_tod if n in summary and summary[n].start_tod is not None else hd.start_tod_sec
        for n, hd in zip(names, heads)]
starts = build_timeline(tods, [hd.duration_sec for hd in heads])
gaps = [(starts[i + 1] - (starts[i] + heads[i].duration_sec)) / 60 for i in range(len(names) - 1)]
print(f"{len(names)} files spanning {(starts[-1] + heads[-1].duration_sec - starts[0]) / 3600:.1f} h of clock time")
print(f"recorded: {sum(hd.duration_sec for hd in heads) / 3600:.1f} h | gaps between files: "
      f"median {np.median(gaps):.1f} min, max {max(gaps):.0f} min")
print("days crossed:", int((starts[-1] - starts[0]) // 86400) + 1)
""")

# ── 2. Preprocessing ─────────────────────────────────────────────────────────
md(r"""
## 2. Preprocessing

`src/preprocess.py`, per subject: 18 bipolar channels → 0.5–40 Hz zero-phase band-pass →
128 Hz → 5-second windows → PLV matrix per window. **Every window is kept, in order, with
its absolute timestamp**, and labelled by time:

| label | rule |
|---|---|
| preictal | window lies within 30 min before an onset |
| ictal | overlaps a seizure (dropped) |
| postictal | within 5 min after a seizure (dropped) |
| excluded | within 60 min of a seizure but none of the above (dropped) |
| interictal | everything else |

Set `RUN_PREPROCESS = True` in the config cell to rebuild `SUBJECT` from the EDF files
(~40 s). The full dataset: `python -m src.preprocess --raw_dir <RAW_DIR> --workers 4` (~5 min).
""")
code(r"""
from src import preprocess

if RUN_PREPROCESS or not (PROC_DIR / SUBJECT / "meta.npz").exists():
    rep = preprocess.preprocess_subject(SUBJECT, str(RAW_DIR), str(PROC_DIR))
else:
    rep = json.load(open(PROC_DIR / SUBJECT / "report.json"))
print(json.dumps({k: rep[k] for k in ["files", "windows", "hours_total", "seizures",
                                      "hours_interictal", "windows_by_label",
                                      "preictal_minutes_per_seizure", "skipped"]}, indent=1))
""")

md(r"""
### The labelled recording
Every 5-second window of the subject's recording, coloured by label. Gaps are time between
files. Note how much interictal data there is: v2 used about 3 hours per subject.
""")
code(r"""
from src.chbmit import LABEL_NAMES

meta = np.load(PROC_DIR / SUBJECT / "meta.npz")
th = (meta["t_start"] - meta["t_start"][0]) / 3600
colors = {0: "#cde2fb", 1: ORANGE, 2: "#e34948", 3: "#9085e9", 4: "#d9d8d3"}
fig, ax = plt.subplots(figsize=(12, 2.2))
for code_, c in colors.items():
    m = meta["label"] == code_
    ax.scatter(th[m], np.zeros(m.sum()), c=c, marker="|", s=300, lw=1,
               label=f"{LABEL_NAMES[code_]} ({m.sum() * 5 / 3600:.1f} h)")
for on, _ in meta["seizures"]:
    ax.axvline((on - meta["t_start"][0]) / 3600, color="#e34948", lw=0.8)
ax.set_yticks([]); ax.set_xlabel("hours from first recording"); ax.grid(False)
ax.legend(ncol=5, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.45))
ax.set_title(f"{SUBJECT}: every window, labelled by time", loc="left")
plt.show()
""")

# ── 3. What the model sees ───────────────────────────────────────────────────
md(r"""
## 3. What the model sees

**Phase Locking Value** between channels *i* and *j* over a window:
PLV = |mean_t exp(i(φᵢ(t) − φⱼ(t)))|, where φ is the Hilbert phase. 0 = no consistent phase
relation, 1 = perfectly locked. The hypothesis: synchrony builds up before a seizure.

Note on montage: a signal identical on every electrode cancels in bipolar channels (A − B),
so PLV here measures synchrony that differs across the scalp, not global common signals.
""")
code(r"""
from src.data import load_subject

d = load_subject(str(PROC_DIR), SUBJECT)
onsets = meta["seizures"][:, 0]
k = int(np.argmax([np.sum(d.block == b) for b in range(len(onsets))]))  # seizure with most preictal data
pre_idx = np.where(d.block == k)[0]
inter_idx = np.where(d.hard == 0)[0]
w_pre, w_int = pre_idx[-1], inter_idx[len(inter_idx) // 2]

fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, (i, name) in zip(axs, [(w_int, "interictal"), (w_pre, "last preictal window")]):
    _, P = d.window(i)
    im = ax.imshow(np.asarray(P, float), vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(18), STANDARD_CHANNELS, rotation=90, fontsize=6)
    ax.set_yticks(range(18), STANDARD_CHANNELS, fontsize=6); ax.grid(False)
    ax.set_title(f"PLV — {name}", loc="left", fontsize=10)
fig.colorbar(im, ax=axs, shrink=0.8)
plt.show()
""")

md("Mean PLV over the 2 hours around that seizure (each dot = one 5-s window):")
code(r"""
iu = np.triu_indices(18, 1)
on = onsets[k]
m = (meta["t_start"] > on - 2 * 3600) & (meta["t_start"] < on + 1800)
rows = np.where(m)[0]
plv_all = np.load(PROC_DIR / SUBJECT / "plv.npy", mmap_mode="r")
mean_plv = np.asarray(plv_all[rows], float)[:, iu[0], iu[1]].mean(1)
tm = (meta["t_start"][rows] - on) / 60
lab = meta["label"][rows]

fig, ax = plt.subplots(figsize=(11, 3.2))
for code_, c in [(0, BLUE), (4, "#b5b4ae"), (1, ORANGE), (2, "#e34948"), (3, "#9085e9")]:
    s = lab == code_
    ax.scatter(tm[s], mean_plv[s], s=3, color=c, label=LABEL_NAMES[code_])
roll = np.convolve(mean_plv, np.ones(36) / 36, mode="same")
ax.plot(tm, roll, color="black", lw=1, label="3-min moving average")
ax.axvline(0, color="#e34948", ls="--"); ax.set_xlabel("minutes from seizure onset"); ax.set_ylabel("mean PLV")
ax.legend(frameon=False, fontsize=8, ncol=6, loc="upper left")
ax.set_title(f"{SUBJECT}, seizure {k + 1}: synchrony around onset", loc="left")
plt.show()
""")

md(r"""
### Soft labels
Instead of 0/1, preictal windows get a target that rises with **time to onset**:
r̃ = 0.10 + 0.90 × (1 − time_to_onset / 30 min). The loss mixes MSE on σ(logit) against r̃
with binary cross-entropy against the 0/1 label (α = 0.5).
""")
code(r"""
sel = d.block == k
tto = (onsets[k] - (d.t_start[sel] + 5)) / 60
fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(-tto, d.risk[sel], color=BLUE, lw=2)
ax.set_xlabel("minutes from onset"); ax.set_ylabel("training target r̃"); ax.set_ylim(0, 1.05)
ax.set_title("Soft-label target across the 30-min preictal period", loc="left")
plt.show()
""")

# ── 4. Model ─────────────────────────────────────────────────────────────────
md(r"""
## 4. Model

Each window becomes a graph: 18 nodes (channels, features = the z-scored 5-s signal), edges
where PLV > 0.3 with PLV as edge weight. A GATv2 branch reads the graph; a temporal
convolution branch reads the channel-averaged signal; the two are fused into one logit.
""")
code(r"""
from torch_geometric.data import Batch
from src.model import STGNN_Soft, window_to_graph

x, P = d.window(w_pre)
x = np.asarray(x, np.float32)
x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
g = window_to_graph(x, np.asarray(P, np.float32), label=1, risk=1.0)
print(g)
print(f"edges kept: {g.num_edges} of {18 * 17} possible")

model = STGNN_Soft().to(DEVICE)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
with torch.no_grad():
    out = model(Batch.from_data_list([g, g]).to(DEVICE))
print("output (logits for a batch of 2):", out.squeeze().tolist())
""")

# ── 5. Splits ────────────────────────────────────────────────────────────────
md(r"""
## 5. Leakage-free splits

* **Leave-one-patient-out** — test on one subject, choose epoch and threshold on 3 other
  subjects, train on the remaining 20.
* **Chronological** — within one subject: first ~50% of the recording to train, next ~20% to
  validate, last ~30% to test. Cuts never split a preictal period.

The asserts below check the two properties that v2 violated.
""")
code(r"""
from src.data import list_subjects
from src.splits import lopo_folds, chronological_split

subjects = list_subjects(str(PROC_DIR))
data = {s: load_subject(str(PROC_DIR), s) for s in subjects}
folds = lopo_folds(subjects, {s: v.n_seizures for s, v in data.items()}, n_val=3, seed=42)
f = next(f for f in folds if f.name == SUBJECT)
print(f"LOPO fold {f.name}: train {len(f.train)} subjects | val {f.val} | test {f.test}")
assert SUBJECT not in f.train and SUBJECT not in f.val

sp = chronological_split(d.block)
for name, idx in sp.items():
    part = d.subset(idx)
    print(f"chrono {name:5s}: {len(idx):6d} windows | {part.n_seizures} seizures | "
          f"{np.sum(part.hard == 0) * 5 / 3600:5.1f} h interictal | "
          f"t = {(part.t_start[0] - d.t_start[0]) / 3600:5.1f}–{(part.t_start[-1] - d.t_start[0]) / 3600:5.1f} h")
assert d.t_start[sp["train"]].max() < d.t_start[sp["val"]].min() < d.t_start[sp["test"]].min()
print("\nOK: test windows come strictly after training windows, and LOPO never sees the test subject.")
""")

# ── 6. Training ──────────────────────────────────────────────────────────────
md(r"""
## 6. Training one fold

Patient-specific (chronological) fold for `SUBJECT`. Each epoch draws 40,000 windows, half
preictal and half interictal; the best epoch is chosen by validation AUC. Then the
logistic-regression baseline (PLV + channel power) is trained on the same split.
""")
code(r"""
from src.train import train_fold, save_fold

OUT = Path("results/notebook")
args = SimpleNamespace(protocol="chrono", epochs=EPOCHS, patience=8, samples_per_epoch=40000,
                       batch_size=128, lr=3e-4, alpha=0.5, pos_weight=1.0, mse_on_logits=False,
                       norm="window", val_max_interictal=20000, num_workers=4, seed=42)
tr, va, te = [d.subset(sp["train"])], [d.subset(sp["val"])], [d.subset(sp["test"])]

t0 = time.time()
model, history, best_auc = train_fold(tr, va, args, DEVICE, AMP)
save_fold(str(OUT / "stgnn"), SUBJECT, model, history, best_auc, va, te, args, DEVICE, AMP)
print(f"done in {time.time() - t0:.0f}s, best validation AUC {best_auc:.3f}")

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot([h["epoch"] for h in history], [h["val_auc"] for h in history], marker="o", color=BLUE)
ax.set_xlabel("epoch"); ax.set_ylabel("validation AUC"); ax.set_title("ST-GNN training", loc="left")
plt.show()
""")
code(r"""
from src import baseline

bargs = SimpleNamespace(max_interictal=20000, C=0.1, seed=42, out_dir=str(OUT / "baseline"))
baseline.run_fold(SUBJECT, tr, va, te, bargs, "chrono")
""")

# ── 7. Evaluation ────────────────────────────────────────────────────────────
md(r"""
## 7. Seizure-level evaluation

Risk scores become **alarms** when 3 of the last 5 windows exceed τ (then 30 min silence).
τ is chosen on the validation part only, targeting ≤ 0.5 false alarms per hour.

* **Seizure sensitivity** — seizures with an alarm in their 30-min preictal period
* **False alarms / hour** — alarms during interictal data ÷ interictal hours
* **Random predictor** — alarms at random with the same rate catch a seizure with
  p = 1 − exp(−FPR × 0.5 h); the binomial p-value tests the model against it

With one subject there are only a few test seizures, so the p-value here cannot be small;
the full runs pool all subjects.
""")
code(r"""
from src.evaluate import evaluate

for name in ["stgnn", "baseline"]:
    evaluate(str(OUT / name))
    display(Markdown(f"#### {name}\n" + (OUT / name / "results.md").read_text()))
""")
code(r"""
from src import figures

figures.main(["--results_dir", str(OUT / "stgnn"), "--subject", SUBJECT])
display(Image(str(OUT / "stgnn" / "figures" / f"risk_timeline_{SUBJECT}.png")))
""")

# ── 8. Full results ──────────────────────────────────────────────────────────
md(r"""
## 8. Full results (from `run_all.sh`)

If the full runs exist under `results/`, this cell collects them. Re-create them with
`bash run_all.sh <RAW_DIR>` or the individual `python -m src.train / src.baseline /
src.evaluate` commands.
""")
code(r"""
runs = [("Baseline", "Leave-one-patient-out", "results/baseline_lopo"),
        ("ST-GNN",   "Leave-one-patient-out", "results/stgnn_lopo"),
        ("Baseline", "Chronological",         "results/baseline_chrono"),
        ("ST-GNN",   "Chronological",         "results/stgnn_chrono")]
lines = ["| Model | Protocol | Seizures predicted | False alarms/h | Random predictor | p vs chance | Window AUC |",
         "|---|---|---|---|---|---|---|"]
for model_name, proto, path in runs:
    p = Path(path) / "results.json"
    if not p.exists():
        lines.append(f"| {model_name} | {proto} | not run yet | | | | |"); continue
    s = json.load(open(p))["summary"]
    pv = "< 0.0001" if s["p_value_vs_chance"] < 1e-4 else f"{s['p_value_vs_chance']:.4f}"
    lines.append(f"| {model_name} | {proto} | {s['predicted']}/{s['seizures']} ({s['sensitivity']:.0%}) | "
                 f"{s['fpr_per_hour']:.2f} | {s['random_predictor_sensitivity']:.0%} | {pv} | "
                 f"{s['window_auc_mean_per_subject']:.2f} ± {s['window_auc_std_per_subject']:.2f} |")
display(Markdown("\n".join(lines)))
""")
code(r"""
for path in ["results/stgnn_lopo/figures/operating_curve.png", "results/stgnn_lopo/figures/per_subject.png"]:
    if Path(path).exists():
        display(Image(path))
""")

# ── 9. Leaky split demo ──────────────────────────────────────────────────────
md(r"""
## 9. Optional: why v2 scored 0.88

The same v3 model and data, but split the way v2 did: windows of several subjects **shuffled
together** and divided 70/15/15. Neighbouring 5-second windows from the same recording now sit
on both sides of the split, so the test set is no longer unseen. Compare the resulting test AUC
with the leave-one-patient-out AUC for the same subjects.

Set `RUN_LEAKY_DEMO = False` in the config cell to skip (takes ~10–20 min on a GPU).
""")
code(r"""
from sklearn.metrics import roc_auc_score
from src.train import predict

if RUN_LEAKY_DEMO:
    demo_subjects = [s for s in subjects if data[s].n_seizures >= 3][:6]
    rng = np.random.default_rng(0)
    tr_l, va_l, te_l = [], [], []
    for s in demo_subjects:
        perm = rng.permutation(len(data[s]))
        a, b = int(0.70 * len(perm)), int(0.85 * len(perm))
        tr_l.append(data[s].subset(np.sort(perm[:a])))
        va_l.append(data[s].subset(np.sort(perm[a:b])))
        te_l.append(data[s].subset(np.sort(perm[b:])))
    largs = SimpleNamespace(**{**vars(args), "epochs": EPOCHS, "protocol": "leaky"})
    leaky_model, _, _ = train_fold(tr_l, va_l, largs, DEVICE, AMP)
    leaky_auc = {p.subject: roc_auc_score(p.hard, predict(leaky_model, p, DEVICE, largs, AMP)) for p in te_l}

    lopo = {}
    if Path("results/stgnn_lopo/results.json").exists():
        lopo = {r["subject"]: r["auc"] for r in json.load(open("results/stgnn_lopo/results.json"))["subjects"]}
    lines = ["| Subject | Random window split (v2-style) | Leave-one-patient-out (v3) |", "|---|---|---|"]
    for s in demo_subjects:
        lines.append(f"| {s} | {leaky_auc[s]:.3f} | {lopo.get(s, float('nan')):.3f} |")
    lines.append(f"| **mean** | **{np.mean(list(leaky_auc.values())):.3f}** | "
                 f"**{np.nanmean([lopo.get(s, np.nan) for s in demo_subjects]):.3f}** |")
    display(Markdown("\n".join(lines)))
else:
    print("skipped (RUN_LEAKY_DEMO = False)")
""")

md(r"""
## Summary

* v3 evaluates on data the model has never seen: other patients, or later recordings.
* Under that protocol the ST-GNN predicts seizures significantly better than chance, with
  modest sensitivity and strong differences between patients.
* Section 9 shows how much of v2's 0.88 came from the split rather than the model.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
os.makedirs("notebooks", exist_ok=True)
with open("notebooks/stgnn_v3_walkthrough.ipynb", "w") as fh:
    json.dump(nb, fh, indent=1)
print(f"wrote notebooks/stgnn_v3_walkthrough.ipynb ({len(cells)} cells)")
