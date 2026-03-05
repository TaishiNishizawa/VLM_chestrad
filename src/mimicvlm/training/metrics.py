# src/mimicvlm/training/metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class MultiLabelMetrics:
    per_label_f1: np.ndarray
    per_label_precision: np.ndarray
    per_label_recall: np.ndarray
    macro_f1: float
    macro_precision: float
    macro_recall: float
    per_label_auroc: Optional[np.ndarray] = None
    macro_auroc: Optional[float] = None


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, np.maximum(den, 1e-12))


def compute_multilabel_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    threshold: float | np.ndarray = 0.5,
    compute_auroc: bool = True,
) -> MultiLabelMetrics:
    """
    logits: (N, C) raw logits
    targets: (N, C) in {0,1} or {0,1,-1}. If -1 exists, those are ignored for F1/AUROC.
    """
    if logits.shape != targets.shape:
        raise ValueError(f"Shape mismatch: logits {logits.shape} vs targets {targets.shape}")

    probs = sigmoid(logits)
    threshold = np.broadcast_to(threshold, (logits.shape[0], logits.shape[1]))
    C = targets.shape[1]

    # Handle unknown labels marked as -1
    mask = targets != -1
    preds = (probs >= threshold).astype(np.int32)

    precisions = np.zeros(C, dtype=np.float64)
    recalls = np.zeros(C, dtype=np.float64)
    f1s = np.zeros(C, dtype=np.float64)

    for c in range(C):
        m = mask[:, c]
        if m.sum() == 0:
            precisions[c] = np.nan
            recalls[c] = np.nan
            f1s[c] = np.nan
            continue

        y = targets[m, c].astype(np.int32)
        p = preds[m, c].astype(np.int32)

        tp = np.sum((p == 1) & (y == 1))
        fp = np.sum((p == 1) & (y == 0))
        fn = np.sum((p == 0) & (y == 1))

        prec = tp / max(tp + fp, 1e-12)
        rec = tp / max(tp + fn, 1e-12)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-12)

        precisions[c] = prec
        recalls[c] = rec
        f1s[c] = f1

    macro_precision = float(np.nanmean(precisions))
    macro_recall = float(np.nanmean(recalls))
    macro_f1 = float(np.nanmean(f1s))

    per_label_auroc = None
    macro_auroc = None

    if compute_auroc:
        try:
            from sklearn.metrics import roc_auc_score  # type: ignore

            aurocs = np.zeros(C, dtype=np.float64)
            for c in range(C):
                m = mask[:, c]
                y = targets[m, c]
                p = probs[m, c]
                # Need both classes present
                if len(np.unique(y)) < 2:
                    aurocs[c] = np.nan
                    continue
                aurocs[c] = roc_auc_score(y, p)

            per_label_auroc = aurocs
            macro_auroc = float(np.nanmean(aurocs))
        except Exception:
            # AUROC will remain None; that's fine for Phase 0
            pass
            
    return MultiLabelMetrics(
        per_label_f1=f1s,
        per_label_precision=precisions,
        per_label_recall=recalls,
        macro_f1=macro_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        per_label_auroc=per_label_auroc,
        macro_auroc=macro_auroc,
    )

def log_per_label_metrics(
    metrics: MultiLabelMetrics,
    label_names: List[str],
) -> None:
    """Pretty-print per-label breakdown. Call once after final eval, not every epoch."""
    C = len(label_names)
    print(f"\n{'Label':35s} {'AUROC':>7} {'F1':>7} {'Prec':>7} {'Recall':>7}")
    print("-" * 65)
    
    valid_auroc = 0
    for c in range(C):
        auroc = metrics.per_label_auroc[c] if metrics.per_label_auroc is not None else np.nan
        f1    = metrics.per_label_f1[c]
        prec  = metrics.per_label_precision[c]
        rec   = metrics.per_label_recall[c]
        
        auroc_str = f"{auroc:.4f}" if not np.isnan(auroc) else "SKIPPED"
        if not np.isnan(auroc):
            valid_auroc += 1
        
        print(f"{label_names[c]:35s} {auroc_str:>7} {f1:>7.4f} {prec:>7.4f} {rec:>7.4f}")
    
    print("-" * 65)
    print(f"{'MACRO':35s} {metrics.macro_auroc:>7.4f} {metrics.macro_f1:>7.4f} "
          f"{metrics.macro_precision:>7.4f} {metrics.macro_recall:>7.4f}")
    print(f"\nAUROC computed over {valid_auroc}/{C} labels")



def find_optimal_thresholds(logits: np.ndarray, targets: np.ndarray,) -> np.ndarray:
    """
    Find per-class threshold maximizing F1. Tune on val, apply on test.
    """
    probs = sigmoid(logits)
    C = targets.shape[1]
    thresholds = np.full(C, 0.5)
    for c in range(C):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.05, 0.95, 0.01):
            preds = (probs[:, c] >= t).astype(np.int32)
            y = targets[:, c].astype(np.int32)
            tp = np.sum((preds == 1) & (y == 1))
            fp = np.sum((preds == 1) & (y == 0))
            fn = np.sum((preds == 0) & (y == 1))
            f1 = (2 * tp) / max(2 * tp + fp + fn, 1e-12)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[c] = best_t
    return thresholds  # shape (C,)


