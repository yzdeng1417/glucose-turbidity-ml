"""Development-only threshold selection and binary growth-score metrics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, confusion_matrix, roc_auc_score

def select_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    y, score = np.asarray(y), np.asarray(score)
    if (y.ndim != 1 or score.shape != y.shape or set(np.unique(y)) != {0, 1}
            or not np.isfinite(score).all() or np.any((score < 0) | (score > 1))):
        raise ValueError("Threshold selection requires finite development scores and both classes")
    # Enumerate every attainable binary decision boundary, including all-negative.
    candidates = np.unique(np.r_[score, 0.0, 0.5, np.nextafter(float(score.max()), np.inf)])
    best = max((float(balanced_accuracy_score(y, score >= threshold)),
                -abs(float(threshold) - 0.5), -float(threshold)) for threshold in candidates)
    return -best[2], best[0]



def metrics(frame: pd.DataFrame, threshold: float | None = None) -> dict:
    truth, score = frame["growth"].to_numpy(), frame["score"].to_numpy()
    both_classes = len(np.unique(truth)) == 2
    result = {"n": len(frame), "growth": int(truth.sum()), "no_growth": int((truth == 0).sum()),
              "roc_auc": float(roc_auc_score(truth, score)) if both_classes else None,
              "average_precision": float(average_precision_score(truth, score)) if truth.sum() else None}
    if threshold is not None:
        predicted = (score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
        result.update({"threshold": threshold, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                       "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)) if both_classes else None,
                       "growth_recall": float(tp / (tp + fn)) if tp + fn else None,
                       "no_growth_recall": float(tn / (tn + fp)) if tn + fp else None})
    return result
