from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def calculate_metrics(
    targets: NDArray[np.int64],
    predictions: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    labels: tuple[str, ...],
) -> dict[str, Any]:
    indices = list(range(len(labels)))
    precision, recall, per_f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=indices,
        zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=indices)
    false_negatives = matrix.sum(axis=1) - np.diag(matrix)
    confidence = np.max(probabilities, axis=1)
    correctness = predictions == targets
    expected_calibration_error = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        mask = (confidence >= lower) & (confidence < lower + 0.1)
        if np.any(mask):
            expected_calibration_error += float(np.mean(mask)) * abs(
                float(np.mean(confidence[mask])) - float(np.mean(correctness[mask]))
            )
    return {
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
        "weighted_f1": float(f1_score(targets, predictions, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(per_f1[index]),
                "support": int(support[index]),
                "false_negatives": int(false_negatives[index]),
            }
            for index, label in enumerate(labels)
        },
        "confusion_matrix": matrix.tolist(),
        "expected_calibration_error_10_bins": expected_calibration_error,
    }
