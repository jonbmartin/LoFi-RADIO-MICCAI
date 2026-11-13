"""
Official LISA 2026 Task 1a metric.

Per artifact, the challenge averages five support-weighted classification metrics (accuracy, F1,
F2, precision, recall) computed over the 0/1/2 severity grades into a per-artifact composite. The
headline score is the mean of that composite over the seven artifacts.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, fbeta_score)

from lisa import ARTIFACTS

# The five metrics the official ranking averages, all "weighted" by class support over the 0/1/2
# grades. Weighted recall == accuracy by construction; both are reported because the spec lists both.
OFFICIAL_METRICS = ["accuracy", "f1", "f2", "precision", "recall"]


def official_severity_metrics(pred: np.ndarray, target: np.ndarray, average: str = "weighted"):
    """Official Task 1a scorer on GRADED severity labels.

    pred, target: integer arrays [N, K] with values in {0,1,2} (K = artifacts).
    Per artifact, computes the five weighted metrics, then the per-artifact 'composite'
    = mean of the five. The headline 'score' is the composite averaged over artifacts
    (matching the challenge's average-of-metrics-then-over-classes ranking).

    Returns dict[artifact] -> {accuracy,f1,f2,precision,recall,composite} plus
    'score' (overall) and 'mean_<metric>' for each of the five.
    """
    pred, target = np.asarray(pred), np.asarray(target)
    out, agg = {}, {m: [] for m in OFFICIAL_METRICS}
    comps = []
    for c, name in enumerate(ARTIFACTS):
        yt, yp = target[:, c], pred[:, c]
        kw = dict(average=average, zero_division=0)
        m = {
            "accuracy": float(accuracy_score(yt, yp)),
            "f1": float(f1_score(yt, yp, **kw)),
            "f2": float(fbeta_score(yt, yp, beta=2.0, **kw)),
            "precision": float(precision_score(yt, yp, **kw)),
            "recall": float(recall_score(yt, yp, **kw)),
        }
        m["composite"] = float(np.mean([m[k] for k in OFFICIAL_METRICS]))
        out[name] = m
        comps.append(m["composite"])
        for k in OFFICIAL_METRICS:
            agg[k].append(m[k])
    for k in OFFICIAL_METRICS:
        out[f"mean_{k}"] = float(np.mean(agg[k]))
    out["score"] = float(np.mean(comps))
    return out


def overcall_diagnostics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Over-calling guardrail: the official support-weighted composite alongside the macro
    composite and the grade-call rate.

      - weighted : the official (support-weighted) score -- what the challenge ranks.
      - macro    : the unweighted-over-classes score.
      - macro_gap: macro - weighted. A large positive gap with a high call_rate is the over-calling
                   signature (good on rare classes, bad on the dominant grade 0).
      - call_rate: fraction of predictions that are grade>=1 (an artifact called). The true rate is
                   ~0.10; a model that calls materially more is trading the dominant grade-0 class
                   for the rare grades, which the support-weighted metric penalizes.
    """
    pred, target = np.asarray(pred), np.asarray(target)
    wt = official_severity_metrics(pred, target, average="weighted")["score"]
    mac = official_severity_metrics(pred, target, average="macro")["score"]
    call = float((pred >= 1).mean())
    true_call = float((target >= 1).mean())
    per_art = {a: float((pred[:, i] >= 1).mean()) for i, a in enumerate(ARTIFACTS)}
    return dict(weighted=wt, macro=mac, macro_gap=mac - wt,
                call_rate=call, true_call_rate=true_call, call_rate_per_artifact=per_art)


def format_official(metrics: dict) -> str:
    cols = f"{'artifact':12s} {'acc':>6s} {'F1':>6s} {'F2':>6s} {'prec':>6s} {'rec':>6s} {'comp':>6s}"
    lines = [cols, "-" * len(cols)]
    for name in ARTIFACTS:
        m = metrics[name]
        lines.append(f"{name:12s} {m['accuracy']:6.3f} {m['f1']:6.3f} {m['f2']:6.3f} "
                     f"{m['precision']:6.3f} {m['recall']:6.3f} {m['composite']:6.3f}")
    lines.append("-" * len(cols))
    lines.append(f"{'MEAN':12s} {metrics['mean_accuracy']:6.3f} {metrics['mean_f1']:6.3f} "
                 f"{metrics['mean_f2']:6.3f} {metrics['mean_precision']:6.3f} "
                 f"{metrics['mean_recall']:6.3f} {metrics['score']:6.3f}")
    return "\n".join(lines)


def present_auroc(present_prob: np.ndarray, target_grades: np.ndarray):
    """Threshold-free sanity check: AUROC of P(any artifact) vs (grade>=1), per artifact.
    Detects classes the model can't rank at all, independent of the severity decision."""
    out, aucs = {}, []
    for c, name in enumerate(ARTIFACTS):
        t = (target_grades[:, c] >= 1).astype(int)
        if 0 < t.sum() < len(t):
            auc = float(roc_auc_score(t, present_prob[:, c])); aucs.append(auc)
        else:
            auc = float("nan")
        out[name] = auc
    out["mean"] = float(np.mean(aucs)) if aucs else float("nan")
    return out
