from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from research.detailed_arrhythmia.evaluation.metrics import calculate_metrics
from research.detailed_arrhythmia.training.model import create_detailed_model


def evaluate_checkpoint_by_record(
    checkpoint_path: Path,
    beats: NDArray[np.float32],
    targets: NDArray[np.int64],
    sources: tuple[str, ...],
    labels: tuple[str, ...],
    *,
    batch_size: int = 1024,
) -> dict[str, dict[str, Any]]:
    model = create_detailed_model(len(labels))
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, beats.shape[0], batch_size):
            logits = model(torch.from_numpy(beats[start : start + batch_size]))
            probabilities.append(torch.softmax(logits, dim=1).numpy())
    probability_array = np.concatenate(probabilities)
    predictions = np.argmax(probability_array, axis=1)
    indices: dict[str, list[int]] = defaultdict(list)
    for index, source in enumerate(sources):
        indices[source].append(index)
    return {
        source: calculate_metrics(
            targets[selected],
            predictions[selected],
            probability_array[selected],
            labels,
        )
        for source, values in sorted(indices.items())
        if (selected := np.asarray(values, dtype=np.int64)).size > 0
    }
