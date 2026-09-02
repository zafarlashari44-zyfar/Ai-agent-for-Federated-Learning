from __future__ import annotations

import math

import numpy as np
import pytest

from reasoning_pipeline.domain.exceptions.pipeline_errors import (
    FeatureExtractionError,
)
from reasoning_pipeline.domain.models.r_peak_series import RPeakSeries
from reasoning_pipeline.scribe_v2.rhythm_feature_extractor import (
    RhythmFeatureExtractor,
)


def create_r_peak_series(
    rr_intervals_ms: tuple[float, ...],
) -> RPeakSeries:
    timestamps = [0.0]

    for interval_ms in rr_intervals_ms:
        timestamps.append(
            timestamps[-1] + interval_ms / 1000.0
        )

    sample_indices = tuple(
        int(round(timestamp * 250.0))
        for timestamp in timestamps
    )

    return RPeakSeries(
        sample_indices=sample_indices,
        timestamps_seconds=tuple(timestamps),
        rr_intervals_ms=rr_intervals_ms,
        detector_name="test_detector",
        detector_version="1.0.0",
        confidence=0.90,
    )


def test_extractor_calculates_regular_rhythm_features() -> None:
    r_peaks = create_r_peak_series(
        (1000.0, 1000.0, 1000.0, 1000.0)
    )

    result = RhythmFeatureExtractor().extract(r_peaks)

    assert result.heart_rate_mean_bpm == pytest.approx(60.0)
    assert result.heart_rate_min_bpm == pytest.approx(60.0)
    assert result.heart_rate_max_bpm == pytest.approx(60.0)
    assert result.mean_rr_ms == pytest.approx(1000.0)
    assert result.sdnn_ms == pytest.approx(0.0)
    assert result.rmssd_ms == pytest.approx(0.0)
    assert result.pnn50_percent == pytest.approx(0.0)
    assert result.irregularity_score == pytest.approx(0.0)


def test_extractor_calculates_variable_heart_rates() -> None:
    r_peaks = create_r_peak_series(
        (1000.0, 750.0, 500.0)
    )

    result = RhythmFeatureExtractor().extract(r_peaks)

    assert result.heart_rate_mean_bpm == pytest.approx(
        (60.0 + 80.0 + 120.0) / 3.0
    )
    assert result.heart_rate_min_bpm == pytest.approx(60.0)
    assert result.heart_rate_max_bpm == pytest.approx(120.0)
    assert result.mean_rr_ms == pytest.approx(750.0)


def test_extractor_uses_sample_sdnn() -> None:
    rr_intervals = (800.0, 900.0, 1000.0)

    result = RhythmFeatureExtractor().extract(
        create_r_peak_series(rr_intervals)
    )

    expected_sdnn = float(
        np.std(
            np.asarray(rr_intervals),
            ddof=1,
        )
    )

    assert result.sdnn_ms == pytest.approx(expected_sdnn)


def test_extractor_calculates_rmssd() -> None:
    result = RhythmFeatureExtractor().extract(
        create_r_peak_series(
            (800.0, 900.0, 800.0)
        )
    )

    assert result.rmssd_ms == pytest.approx(100.0)


def test_extractor_calculates_pnn50() -> None:
    result = RhythmFeatureExtractor().extract(
        create_r_peak_series(
            (800.0, 900.0, 930.0, 850.0)
        )
    )

    assert result.pnn50_percent == pytest.approx(
        2.0 / 3.0 * 100.0
    )


def test_regular_rhythm_has_lower_irregularity_than_variable_rhythm() -> None:
    extractor = RhythmFeatureExtractor()

    regular = extractor.extract(
        create_r_peak_series(
            (1000.0, 1000.0, 1000.0, 1000.0)
        )
    )

    variable = extractor.extract(
        create_r_peak_series(
            (600.0, 1200.0, 650.0, 1300.0)
        )
    )

    assert regular.irregularity_score is not None
    assert variable.irregularity_score is not None
    assert variable.irregularity_score > regular.irregularity_score


def test_irregularity_score_is_bounded() -> None:
    result = RhythmFeatureExtractor().extract(
        create_r_peak_series(
            (300.0, 2000.0, 350.0, 2200.0)
        )
    )

    assert result.irregularity_score is not None
    assert 0.0 <= result.irregularity_score <= 1.0


def test_single_rr_interval_returns_basic_features() -> None:
    result = RhythmFeatureExtractor().extract(
        create_r_peak_series((1000.0,))
    )

    assert result.heart_rate_mean_bpm == pytest.approx(60.0)
    assert result.heart_rate_min_bpm == pytest.approx(60.0)
    assert result.heart_rate_max_bpm == pytest.approx(60.0)
    assert result.mean_rr_ms == pytest.approx(1000.0)
    assert result.sdnn_ms is None
    assert result.rmssd_ms is None
    assert result.pnn50_percent is None
    assert result.irregularity_score is None


def test_empty_series_returns_empty_features() -> None:
    r_peaks = RPeakSeries(
        sample_indices=(),
        timestamps_seconds=(),
        rr_intervals_ms=(),
        detector_name="test_detector",
        detector_version="1.0.0",
        confidence=0.0,
    )

    result = RhythmFeatureExtractor().extract(r_peaks)

    assert result.heart_rate_mean_bpm is None
    assert result.heart_rate_min_bpm is None
    assert result.heart_rate_max_bpm is None
    assert result.mean_rr_ms is None
    assert result.sdnn_ms is None
    assert result.rmssd_ms is None
    assert result.pnn50_percent is None
    assert result.irregularity_score is None


def test_extractor_rejects_rr_count_mismatch() -> None:
    r_peaks = RPeakSeries(
        sample_indices=(100, 200, 300),
        timestamps_seconds=(0.4, 0.8, 1.2),
        rr_intervals_ms=(400.0,),
        detector_name="test_detector",
        detector_version="1.0.0",
        confidence=0.8,
    )

    with pytest.raises(
        FeatureExtractionError,
        match="RR interval count",
    ):
        RhythmFeatureExtractor().extract(r_peaks)


def test_extractor_rejects_rr_below_minimum() -> None:
    r_peaks = create_r_peak_series(
        (200.0, 1000.0)
    )

    with pytest.raises(
        FeatureExtractionError,
        match="below the configured physiological minimum",
    ):
        RhythmFeatureExtractor().extract(r_peaks)


def test_extractor_accepts_long_rr_interval() -> None:
    r_peaks = create_r_peak_series(
        (1000.0, 3000.0)
    )

    result = RhythmFeatureExtractor().extract(r_peaks)

    assert result.mean_rr_ms == pytest.approx(2000.0)
    assert result.heart_rate_min_bpm == pytest.approx(20.0)

def test_extractor_rejects_non_increasing_sample_indices() -> None:
    r_peaks = RPeakSeries(
        sample_indices=(100, 100, 350),
        timestamps_seconds=(0.4, 0.8, 1.4),
        rr_intervals_ms=(400.0, 600.0),
        detector_name="test_detector",
        detector_version="1.0.0",
        confidence=0.8,
    )

    with pytest.raises(
        FeatureExtractionError,
        match="sample indices must be strictly increasing",
    ):
        RhythmFeatureExtractor().extract(r_peaks)


def test_extractor_rejects_non_increasing_timestamps() -> None:
    r_peaks = RPeakSeries(
        sample_indices=(100, 200, 350),
        timestamps_seconds=(0.4, 0.4, 1.4),
        rr_intervals_ms=(400.0, 600.0),
        detector_name="test_detector",
        detector_version="1.0.0",
        confidence=0.8,
    )

    with pytest.raises(
        FeatureExtractionError,
        match="timestamps must be strictly increasing",
    ):
        RhythmFeatureExtractor().extract(r_peaks)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {"minimum_rr_ms": 0.0},
            "minimum_rr_ms must be greater than zero",
        ),
        (
            {
                "minimum_rr_ms": 1000.0,
                "maximum_rr_ms": 500.0,
            },
            "maximum_rr_ms must be greater",
        ),
        (
            {"cv_reference": 0.0},
            "cv_reference must be greater than zero",
        ),
        (
            {"rmssd_reference_ratio": 0.0},
            "rmssd_reference_ratio must be greater than zero",
        ),
        (
            {"pnn50_reference_percent": 0.0},
            "pnn50_reference_percent must be greater than zero",
        ),
        (
            {"cv_reference": math.inf},
            "configuration must be finite",
        ),
    ],
)
def test_extractor_rejects_invalid_configuration(
    arguments: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        RhythmFeatureExtractor(**arguments)


def test_output_is_immutable() -> None:
    result = RhythmFeatureExtractor().extract(
        create_r_peak_series(
            (1000.0, 1000.0, 1000.0)
        )
    )

    with pytest.raises(AttributeError):
        result.mean_rr_ms = 500.0  # type: ignore[misc]