"""Window-level and seizure-level (event) metrics for seizure prediction.

Window metrics (AUC, sensitivity, specificity, F1) score each 5-second window on its own.
They are useful for model selection but they are *not* how a prediction system is judged.

Event metrics follow the standard seizure-prediction protocol:

* A per-window risk ``r_t`` becomes an **alarm** when at least ``k`` of the last ``n``
  windows in the same contiguous recording exceed ``tau`` (suppresses one-window spikes).
  After an alarm, further alarms are suppressed for a **refractory** period.
* A seizure counts as **predicted** if an alarm fires inside its preictal block (the
  seizure occurrence period, SOP, before onset).
* **Lead time** is measured per seizure: from the first alarm to the end of that seizure's
  preictal block. It is never summed across seizures.
* **False alarms per hour (FPR/h)** = alarms raised in interictal windows divided by the
  hours of interictal data evaluated.
* A **random predictor** with the same FPR/h raises an alarm in any SOP with probability
  ``p = 1 - exp(-FPR/h * SOP_h)`` (Schelter et al., 2006). The p-value is the binomial
  probability of predicting at least as many seizures by chance.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom
from sklearn.metrics import f1_score, roc_auc_score


# ── Window level ─────────────────────────────────────────────────────────────

def window_metrics(probs: np.ndarray, hard: np.ndarray, tau: float) -> dict:
    probs, hard = np.asarray(probs), np.asarray(hard)
    pred = probs >= tau
    tp = int(np.sum(pred & (hard == 1)))
    fn = int(np.sum(~pred & (hard == 1)))
    tn = int(np.sum(~pred & (hard == 0)))
    fp = int(np.sum(pred & (hard == 0)))
    auc = roc_auc_score(hard, probs) if len(np.unique(hard)) > 1 else float("nan")
    return {
        "auc": float(auc),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "f1": float(f1_score(hard, pred, zero_division=0)),
    }


# ── Alarms ───────────────────────────────────────────────────────────────────

def raise_alarms(probs: np.ndarray, run: np.ndarray, tau: float, k: int = 3, n: int = 5,
                 refractory: int = 360) -> np.ndarray:
    """Boolean array marking the windows where an alarm fires.

    ``k``-of-``n`` smoothing only looks back within the same contiguous run, so windows on
    either side of a recording gap never vote together. ``refractory`` is in windows
    (360 x 5 s = 30 min, matching the SOP).
    """
    probs, run = np.asarray(probs), np.asarray(run)
    above = (probs >= tau).astype(np.int64)
    alarms = np.zeros(len(probs), dtype=bool)
    last = -10**12
    run_start = 0
    for i in range(len(probs)):
        if i > 0 and run[i] != run[i - 1]:
            run_start = i
        lo = max(run_start, i - n + 1)
        if above[lo:i + 1].sum() >= k and i - last >= refractory:
            alarms[i] = True
            last = i
    return alarms


# ── Seizure level ────────────────────────────────────────────────────────────

def event_metrics(alarms: np.ndarray, hard: np.ndarray, block: np.ndarray,
                  window_sec: float = 5.0, min_preictal_windows: int = 1) -> dict:
    """Per-subject event metrics. Windows must be in recording order.

    Seizures with fewer than ``min_preictal_windows`` preictal windows recorded (e.g. a
    seizure minutes after a recording starts) cannot fairly be predicted; they are left
    out of the seizure count, and alarms inside them count neither way.
    """
    alarms, hard, block = np.asarray(alarms), np.asarray(hard), np.asarray(block)
    seizures, skipped = [], 0
    for b in np.unique(block[block >= 0]):
        idx = np.where(block == b)[0]
        if len(idx) < min_preictal_windows:
            skipped += 1
            continue
        hits = idx[alarms[idx]]
        lead = (idx.max() - hits.min() + 1) * window_sec / 60 if len(hits) else None
        seizures.append({
            "block": int(b),
            "predicted": bool(len(hits)),
            "lead_time_min": lead,
            "preictal_min": len(idx) * window_sec / 60,
        })
    false_alarms = int(np.sum(alarms & (hard == 0)))
    interictal_h = float(np.sum(hard == 0) * window_sec / 3600)
    return {
        "n_seizures": len(seizures),
        "n_predicted": sum(s["predicted"] for s in seizures),
        "false_alarms": false_alarms,
        "interictal_hours": interictal_h,
        "seizures_skipped_short_preictal": skipped,
        "seizures": seizures,
    }


def summarize_events(per_subject: list[dict], sop_min: float = 30.0) -> dict:
    """Pool event counts across subjects and compare against a random predictor."""
    n_sz = sum(r["n_seizures"] for r in per_subject)
    n_pred = sum(r["n_predicted"] for r in per_subject)
    fa = sum(r["false_alarms"] for r in per_subject)
    hours = sum(r["interictal_hours"] for r in per_subject)
    leads = [s["lead_time_min"] for r in per_subject for s in r["seizures"] if s["predicted"]]

    sens = n_pred / n_sz if n_sz else float("nan")
    fpr_h = fa / hours if hours else float("nan")
    p_chance = 1 - np.exp(-fpr_h * sop_min / 60) if hours else float("nan")
    p_value = float(binom.sf(n_pred - 1, n_sz, p_chance)) if n_sz else float("nan")

    return {
        "seizures": n_sz,
        "seizures_skipped_short_preictal": sum(r.get("seizures_skipped_short_preictal", 0) for r in per_subject),
        "predicted": n_pred,
        "sensitivity": sens,
        "false_alarms": fa,
        "interictal_hours": hours,
        "fpr_per_hour": fpr_h,
        "lead_time_mean_min": float(np.mean(leads)) if leads else float("nan"),
        "lead_time_median_min": float(np.median(leads)) if leads else float("nan"),
        "random_predictor_sensitivity": float(p_chance),
        "p_value_vs_chance": p_value,
        "beats_chance_at_0.05": bool(p_value < 0.05),
    }


# ── Threshold selection (validation data only) ───────────────────────────────

def select_threshold(val_sets: list[dict], mode: str = "fpr", target_fpr: float = 0.5,
                     k: int = 3, n: int = 5, refractory: int = 360,
                     window_sec: float = 5.0, grid=None, min_preictal_windows: int = 1) -> float:
    """Pick ``tau`` on validation subjects only.

    ``val_sets`` holds one dict per validation subject with keys
    ``probs``, ``hard``, ``block``, ``run`` (all in recording order).

    mode="f1"  : maximise window-level F1 (what the notebook did).
    mode="fpr" : maximise seizure sensitivity subject to FPR/h <= ``target_fpr``;
                 if no threshold meets the target, return the one with the lowest FPR/h.
    """
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2) if grid is None else grid
    if mode == "f1":
        probs = np.concatenate([v["probs"] for v in val_sets])
        hard = np.concatenate([v["hard"] for v in val_sets])
        scores = [f1_score(hard, probs >= t, zero_division=0) for t in grid]
        return float(grid[int(np.argmax(scores))])

    best = None  # (meets_target, sensitivity, -fpr, tau)
    for t in grid:
        per = [event_metrics(raise_alarms(v["probs"], v["run"], t, k, n, refractory),
                             v["hard"], v["block"], window_sec, min_preictal_windows) for v in val_sets]
        s = summarize_events(per)
        sens = 0.0 if np.isnan(s["sensitivity"]) else s["sensitivity"]
        key = (s["fpr_per_hour"] <= target_fpr, sens if s["fpr_per_hour"] <= target_fpr else 0.0,
               -s["fpr_per_hour"], -t)
        if best is None or key > best[0]:
            best = (key, float(t))
    return best[1]
