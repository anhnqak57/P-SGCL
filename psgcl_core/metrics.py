from __future__ import annotations

import numpy as np
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize


METRIC_NAMES = (
    "precision", "recall", "f1", "f1_weighted", "acc", "auc", "prauc",
    "ari", "mcc", "dbi", "ss",
)


def _nan_if_invalid(callable_):
    try:
        return float(callable_())
    except ValueError:
        return float("nan")


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    embedding: np.ndarray,
    n_classes: int,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=float)
    one_hot = label_binarize(y_true, classes=np.arange(n_classes))
    if n_classes == 2:
        one_hot = np.column_stack([1 - one_hot.ravel(), one_hot.ravel()])

    auc = _nan_if_invalid(lambda: metrics.roc_auc_score(one_hot, y_prob, average="macro", multi_class="ovr"))
    prauc = _nan_if_invalid(
        lambda: metrics.average_precision_score(one_hot, y_prob, average="macro")
    )
    if n_classes == 2:

        auc = _nan_if_invalid(lambda: metrics.roc_auc_score(one_hot.ravel(), y_prob.ravel()))
        precision_curve, recall_curve, _ = precision_recall_curve(one_hot.ravel(), y_prob.ravel())
        prauc = _nan_if_invalid(lambda: metrics.auc(recall_curve, precision_curve))

    return {
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "auc": auc,
        "prauc": prauc,
        "ari": float(metrics.adjusted_rand_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "dbi": _nan_if_invalid(lambda: metrics.davies_bouldin_score(embedding, y_true)),
        "ss": _nan_if_invalid(lambda: metrics.silhouette_score(embedding, y_true)),
    }


def summarise(metrics_by_fold: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not metrics_by_fold:
        raise ValueError("No folds were evaluated.")
    summary: dict[str, dict[str, float]] = {}
    for name in METRIC_NAMES:
        values = np.asarray([row[name] for row in metrics_by_fold], dtype=float)
        summary[name] = {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values))}
    return summary
