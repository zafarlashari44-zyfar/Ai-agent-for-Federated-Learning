from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def normalise_confusion_matrix(
    matrix: NDArray[np.int64],
) -> NDArray[np.float64]:
    row_totals = matrix.sum(axis=1, keepdims=True)

    return np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    )


def analyse_confusion_matrix(
    matrix: NDArray[np.int64],
    labels: tuple[str, ...],
    normal_label: str = "N",
) -> dict[str, Any]:
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError(
            "Confusion matrix dimensions do not match ontology labels"
        )

    normalised = normalise_confusion_matrix(matrix)

    normal_index = labels.index(normal_label)

    per_class: dict[str, Any] = {}

    for true_index, true_label in enumerate(labels):
        row = matrix[true_index]
        normalised_row = normalised[true_index]

        support = int(row.sum())
        correct = int(row[true_index])

        errors: list[dict[str, Any]] = []

        for predicted_index, predicted_label in enumerate(labels):
            if predicted_index == true_index:
                continue

            count = int(row[predicted_index])

            errors.append(
                {
                    "predicted_label": predicted_label,
                    "count": count,
                    "fraction": float(normalised_row[predicted_index]),
                }
            )

        errors.sort(
            key=lambda item: item["count"],
            reverse=True,
        )

        abnormal_to_normal = 0
        abnormal_to_normal_fraction = 0.0

        if true_label != normal_label:
            abnormal_to_normal = int(row[normal_index])

            if support > 0:
                abnormal_to_normal_fraction = (
                    abnormal_to_normal / support
                )

        per_class[true_label] = {
            "support": support,
            "correct": correct,
            "recall_from_matrix": (
                float(correct / support)
                if support > 0
                else 0.0
            ),
            "predicted_distribution": {
                predicted_label: {
                    "count": int(row[predicted_index]),
                    "fraction": float(
                        normalised_row[predicted_index]
                    ),
                }
                for predicted_index, predicted_label in enumerate(labels)
            },
            "top_errors": errors,
            "abnormal_to_normal_count": abnormal_to_normal,
            "abnormal_to_normal_fraction": abnormal_to_normal_fraction,
        }

    abnormal_to_normal_total = sum(
        details["abnormal_to_normal_count"]
        for label, details in per_class.items()
        if label != normal_label
    )

    abnormal_support = sum(
        details["support"]
        for label, details in per_class.items()
        if label != normal_label
    )

    return {
        "labels": list(labels),
        "raw_confusion_matrix": matrix.tolist(),
        "row_normalised_confusion_matrix": normalised.tolist(),
        "per_class": per_class,
        "overall_abnormal_to_normal_count": abnormal_to_normal_total,
        "overall_abnormal_support": abnormal_support,
        "overall_abnormal_to_normal_fraction": (
            abnormal_to_normal_total / abnormal_support
            if abnormal_support > 0
            else 0.0
        ),
    }


def aggregate_confusion_matrices(
    matrices: list[NDArray[np.int64]],
) -> NDArray[np.int64]:
    if not matrices:
        raise ValueError("At least one confusion matrix is required")

    first_shape = matrices[0].shape

    if any(matrix.shape != first_shape for matrix in matrices):
        raise ValueError(
            "All confusion matrices must have the same dimensions"
        )

    return np.sum(
        np.stack(matrices, axis=0),
        axis=0,
        dtype=np.int64,
    )