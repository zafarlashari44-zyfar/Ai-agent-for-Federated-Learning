from typing import Literal

import numpy as np
from numpy.typing import NDArray

WeightMethod = Literal[
    "inverse_frequency",
    "sqrt_inverse_frequency",
    "effective_number",
]


def calculate_class_weights(
    training_labels: NDArray[np.int64],
    num_classes: int,
    *,
    method: WeightMethod = "sqrt_inverse_frequency",
    effective_number_beta: float = 0.9999,
) -> NDArray[np.float32]:
    counts = np.bincount(training_labels, minlength=num_classes).astype(np.float64)
    if np.any(counts <= 0):
        raise ValueError("Every configured class must occur in the training split")
    if method == "inverse_frequency":
        weights = 1.0 / counts
    elif method == "sqrt_inverse_frequency":
        weights = 1.0 / np.sqrt(counts)
    elif method == "effective_number":
        if not 0.0 < effective_number_beta < 1.0:
            raise ValueError("effective_number_beta must be between zero and one")
        weights = (1.0 - effective_number_beta) / (
            1.0 - np.power(effective_number_beta, counts)
        )
    else:
        raise ValueError(f"Unsupported class-weight method: {method}")
    weights = weights / np.mean(weights)
    return np.asarray(weights, dtype=np.float32)
