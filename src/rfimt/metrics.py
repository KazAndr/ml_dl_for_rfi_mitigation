"""Metrics and threshold helpers for binary channel classification."""

from __future__ import annotations

import numpy as np


def best_threshold_by_f1(y_true, scores, grid=None) -> tuple[float, float]:
    """Select the threshold with the best F1 score."""
    from sklearn.metrics import precision_recall_fscore_support

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)

    best_t, best_f1 = 0.5, -1.0
    for threshold in grid:
        pred = (scores >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true,
            pred,
            average="binary",
            zero_division=0,
        )
        if f1 > best_f1:
            best_t = float(threshold)
            best_f1 = float(f1)
    return best_t, best_f1


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    """Return TN, FP, FN and TP as a dictionary."""
    from sklearn.metrics import confusion_matrix

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def eval_binary(y_true, scores, threshold: float = 0.5) -> dict[str, float]:
    """Evaluate binary scores at one threshold."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        log_loss,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    pred = (scores >= threshold).astype(int)
    has_both_classes = len(np.unique(y_true)) == 2

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        pred,
        average="binary",
        zero_division=0,
    )
    out = {
        **confusion_counts(y_true, pred),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, scores)) if has_both_classes else np.nan,
        "pr_auc": float(average_precision_score(y_true, scores)) if has_both_classes else np.nan,
        "logloss": (
            float(log_loss(y_true, np.c_[1 - scores, scores], labels=[0, 1]))
            if has_both_classes
            else np.nan
        ),
    }
    return out
