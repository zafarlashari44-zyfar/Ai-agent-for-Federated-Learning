from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr


INPUT_LENGTH = 216
COMMON_REGIONS = 54
SAMPLES_PER_REGION = INPUT_LENGTH // COMMON_REGIONS


@dataclass(frozen=True)
class AttributionComparison:
    pearson: float
    spearman: float

    top_10_overlap: float
    top_10_iou: float

    top_20_overlap: float
    top_20_iou: float

    gradcam_peak_region: int
    shap_peak_region: int

    peak_distance_regions: int
    peak_distance_samples: int
    peak_distance_ms: float


def aggregate_to_regions(
    values: np.ndarray,
    *,
    absolute: bool = False,
) -> np.ndarray:
    """Aggregate 216 sample attributions into 54 temporal regions."""

    array = np.asarray(
        values,
        dtype=np.float64,
    ).reshape(-1)

    if array.shape != (INPUT_LENGTH,):
        raise ValueError(
            f"Expected {INPUT_LENGTH} attribution values, "
            f"received {array.shape}"
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            "Attribution contains non finite values"
        )

    if absolute:
        array = np.abs(array)

    regions = array.reshape(
        COMMON_REGIONS,
        SAMPLES_PER_REGION,
    )

    return regions.mean(axis=1)


def normalise_importance(
    values: np.ndarray,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    maximum = float(array.max())

    if maximum <= 0:
        return np.zeros_like(array)

    return array / maximum


def top_region_indices(
    values: np.ndarray,
    fraction: float,
) -> set[int]:
    k = max(
        1,
        int(round(len(values) * fraction)),
    )

    indices = np.argsort(values)[-k:]

    return set(
        int(index)
        for index in indices
    )


def calculate_overlap(
    first: np.ndarray,
    second: np.ndarray,
    fraction: float,
) -> tuple[float, float]:
    first_top = top_region_indices(
        first,
        fraction,
    )

    second_top = top_region_indices(
        second,
        fraction,
    )

    intersection = len(
        first_top & second_top
    )

    union = len(
        first_top | second_top
    )

    overlap = (
        intersection / len(first_top)
        if first_top
        else 0.0
    )

    iou = (
        intersection / union
        if union
        else 0.0
    )

    return overlap, iou


def compare_gradcam_shap(
    *,
    gradcam_values: np.ndarray,
    shap_values: np.ndarray,
    sampling_rate_hz: float,
) -> AttributionComparison:
    """
    Compare GradCAM and SHAP at a common 54 region resolution.

    GradCAM is treated as unsigned importance.

    SHAP is converted to absolute importance for the agreement
    comparison. Signed SHAP should be retained separately for
    interpretation.
    """

    if sampling_rate_hz <= 0:
        raise ValueError(
            "sampling_rate_hz must be positive"
        )

    gradcam_regions = aggregate_to_regions(
        gradcam_values,
        absolute=False,
    )

    shap_regions = aggregate_to_regions(
        shap_values,
        absolute=True,
    )

    gradcam_regions = normalise_importance(
        gradcam_regions
    )

    shap_regions = normalise_importance(
        shap_regions
    )

    pearson_result = pearsonr(
        gradcam_regions,
        shap_regions,
    )

    spearman_result = spearmanr(
        gradcam_regions,
        shap_regions,
    )

    top_10_overlap, top_10_iou = calculate_overlap(
        gradcam_regions,
        shap_regions,
        0.10,
    )

    top_20_overlap, top_20_iou = calculate_overlap(
        gradcam_regions,
        shap_regions,
        0.20,
    )

    gradcam_peak_region = int(
        np.argmax(gradcam_regions)
    )

    shap_peak_region = int(
        np.argmax(shap_regions)
    )

    peak_distance_regions = abs(
        gradcam_peak_region
        - shap_peak_region
    )

    peak_distance_samples = (
        peak_distance_regions
        * SAMPLES_PER_REGION
    )

    peak_distance_ms = (
        peak_distance_samples
        / sampling_rate_hz
        * 1000.0
    )

    return AttributionComparison(
        pearson=float(
            pearson_result.statistic
        ),
        spearman=float(
            spearman_result.statistic
        ),
        top_10_overlap=top_10_overlap,
        top_10_iou=top_10_iou,
        top_20_overlap=top_20_overlap,
        top_20_iou=top_20_iou,
        gradcam_peak_region=gradcam_peak_region,
        shap_peak_region=shap_peak_region,
        peak_distance_regions=peak_distance_regions,
        peak_distance_samples=peak_distance_samples,
        peak_distance_ms=peak_distance_ms,
    )