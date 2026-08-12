from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from research.detailed_arrhythmia.dataset.annotations import patient_id_for_record
from research.detailed_arrhythmia.evaluation.metrics import calculate_metrics
from research.detailed_arrhythmia.training.model import create_detailed_model


def _predict_checkpoint(
    checkpoint_path: Path,
    beats: NDArray[np.float32],
    labels: tuple[str, ...],
    *,
    batch_size: int = 1024,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    model = create_detailed_model(len(labels))

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state)
    model.eval()

    probabilities: list[NDArray[np.float64]] = []

    with torch.no_grad():
        for start in range(0, beats.shape[0], batch_size):
            batch = torch.from_numpy(
                beats[start : start + batch_size]
            )

            logits = model(batch)

            batch_probabilities = (
                torch.softmax(logits, dim=1)
                .numpy()
                .astype(np.float64)
            )

            probabilities.append(batch_probabilities)

    probability_array = np.concatenate(probabilities, axis=0)

    predictions = np.argmax(
        probability_array,
        axis=1,
    ).astype(np.int64)

    return predictions, probability_array


def _evaluate_by_groups(
    targets: NDArray[np.int64],
    predictions: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    groups: tuple[str, ...],
    labels: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    indices: dict[str, list[int]] = defaultdict(list)

    for index, group in enumerate(groups):
        indices[group].append(index)

    output: dict[str, dict[str, Any]] = {}

    normal_index = labels.index("N")

    for group, values in sorted(indices.items()):
        selected = np.asarray(values, dtype=np.int64)

        group_targets = targets[selected]
        group_predictions = predictions[selected]
        group_probabilities = probabilities[selected]

        metrics = calculate_metrics(
            group_targets,
            group_predictions,
            group_probabilities,
            labels,
        )

        present_labels = [
        label
        for label in labels
        if metrics["per_class"][label]["support"] > 0
        ]

        present_class_f1_values = [
            metrics["per_class"][label]["f1"]
            for label in present_labels
        ]

        metrics["present_labels"] = present_labels

        metrics["macro_f1_present_classes"] = (
            float(np.mean(present_class_f1_values))
            if present_class_f1_values
            else 0.0
        )

        abnormal_mask = group_targets != normal_index
        abnormal_support = int(np.sum(abnormal_mask))

        abnormal_to_normal = int(
            np.sum(
                abnormal_mask
                & (group_predictions == normal_index)
            )
        )

        metrics["total_beats"] = int(selected.size)
        metrics["abnormal_support"] = abnormal_support
        metrics["abnormal_to_normal_count"] = abnormal_to_normal

        metrics["abnormal_to_normal_fraction"] = (
            abnormal_to_normal / abnormal_support
            if abnormal_support > 0
            else 0.0
        )

        output[group] = metrics

    return output


def evaluate_checkpoint_by_record(
    checkpoint_path: Path,
    beats: NDArray[np.float32],
    targets: NDArray[np.int64],
    sources: tuple[str, ...],
    labels: tuple[str, ...],
    *,
    batch_size: int = 1024,
) -> dict[str, dict[str, Any]]:
    predictions, probabilities = _predict_checkpoint(
        checkpoint_path,
        beats,
        labels,
        batch_size=batch_size,
    )

    return _evaluate_by_groups(
        targets,
        predictions,
        probabilities,
        sources,
        labels,
    )


def evaluate_checkpoint_by_patient(
    checkpoint_path: Path,
    beats: NDArray[np.float32],
    targets: NDArray[np.int64],
    sources: tuple[str, ...],
    labels: tuple[str, ...],
    *,
    batch_size: int = 1024,
) -> dict[str, dict[str, Any]]:
    predictions, probabilities = _predict_checkpoint(
        checkpoint_path,
        beats,
        labels,
        batch_size=batch_size,
    )

    patient_sources = tuple(
        patient_id_for_record(record_id)
        for record_id in sources
    )

    return _evaluate_by_groups(
        targets,
        predictions,
        probabilities,
        patient_sources,
        labels,
    )