from __future__ import annotations

import numpy as np
import pytest

from reasoning_pipeline.domain.enums.statuses import SignalQualityStatus
from reasoning_pipeline.domain.exceptions.pipeline_errors import (
    FeatureExtractionError,
    InsufficientSignalQualityError,
    UnsupportedSamplingRateError,
)
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.domain.models.signal_quality import SignalQuality
from reasoning_pipeline.scribe_v2.r_peak_detector import (
    SciPyRPeakDetector,
)


def create_signal(
    samples: np.ndarray,
    *,
    sampling_rate_hz: float = 250.0,
) -> ECGSignal:
    return ECGSignal(
        record_id="synthetic-record",
        samples=tuple(float(value) for value in samples),
        sampling_rate_hz=sampling_rate_hz,
        source="synthetic",
        lead_name="Lead II",
    )


def create_synthetic_ecg(
    *,
    sampling_rate_hz: float = 250.0,
    duration_seconds: float = 10.0,
    heart_rate_bpm: float = 60.0,
    inverted: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    sample_count = int(
        round(sampling_rate_hz * duration_seconds)
    )

    time = (
        np.arange(sample_count, dtype=np.float64)
        / sampling_rate_hz
    )

    signal = 0.03 * np.sin(2.0 * np.pi * 0.4 * time)

    rr_seconds = 60.0 / heart_rate_bpm

    peak_times = np.arange(
        0.75,
        duration_seconds - 0.25,
        rr_seconds,
        dtype=np.float64,
    )

    expected_indices = np.asarray(
        np.round(peak_times * sampling_rate_hz),
        dtype=np.int64,
    )

    qrs_width_seconds = 0.025
    polarity = -1.0 if inverted else 1.0

    for peak_time in peak_times:
        signal += polarity * np.exp(
            -0.5
            * (
                (time - peak_time)
                / qrs_width_seconds
            )
            ** 2
        )

    return signal, expected_indices


def test_detector_detects_synthetic_r_peaks() -> None:
    samples, expected_indices = create_synthetic_ecg()

    result = SciPyRPeakDetector().detect(
        create_signal(samples)
    )

    assert result.peak_count == expected_indices.size
    assert len(result.rr_intervals_ms) == result.peak_count - 1
    assert result.detector_name == "scipy_adaptive_r_peak"
    assert result.detector_version == "1.1.0"
    assert 0.0 <= result.confidence <= 1.0

    np.testing.assert_allclose(
        np.asarray(result.sample_indices),
        expected_indices,
        atol=3,
    )


def test_detector_calculates_rr_intervals() -> None:
    samples, _ = create_synthetic_ecg(
        heart_rate_bpm=60.0,
    )

    result = SciPyRPeakDetector().detect(
        create_signal(samples)
    )

    np.testing.assert_allclose(
        np.asarray(result.rr_intervals_ms),
        1000.0,
        atol=20.0,
    )


def test_detector_creates_peak_timestamps() -> None:
    samples, _ = create_synthetic_ecg()

    result = SciPyRPeakDetector().detect(
        create_signal(samples)
    )

    expected_timestamps = (
        np.asarray(result.sample_indices, dtype=np.float64)
        / 250.0
    )

    np.testing.assert_allclose(
        np.asarray(result.timestamps_seconds),
        expected_timestamps,
    )


def test_detector_handles_inverted_r_peaks() -> None:
    samples, expected_indices = create_synthetic_ecg(
        inverted=True,
    )

    result = SciPyRPeakDetector().detect(
        create_signal(samples)
    )

    assert result.peak_count == expected_indices.size

    np.testing.assert_allclose(
        np.asarray(result.sample_indices),
        expected_indices,
        atol=3,
    )


def test_detector_accepts_non_unusable_quality() -> None:
    samples, _ = create_synthetic_ecg()

    quality = SignalQuality(
        score=0.85,
        status=SignalQualityStatus.GOOD,
        noise_score=0.10,
        valid_sample_ratio=1.0,
    )

    result = SciPyRPeakDetector().detect(
        create_signal(samples),
        signal_quality=quality,
    )

    assert result.peak_count >= 2
    assert result.confidence > 0.0


def test_detector_rejects_unusable_signal_quality() -> None:
    samples, _ = create_synthetic_ecg()

    quality = SignalQuality(
        score=0.10,
        status=SignalQualityStatus.UNUSABLE,
        noise_score=0.95,
        valid_sample_ratio=1.0,
    )

    with pytest.raises(
        InsufficientSignalQualityError,
        match="unusable ECG signal",
    ):
        SciPyRPeakDetector().detect(
            create_signal(samples),
            signal_quality=quality,
        )


def test_detector_rejects_constant_signal() -> None:
    samples = np.ones(2500, dtype=np.float64)

    with pytest.raises(
        FeatureExtractionError,
        match="negligible amplitude variation",
    ):
        SciPyRPeakDetector().detect(
            create_signal(samples)
        )


def test_detector_rejects_signal_with_non_finite_values() -> None:
    samples, _ = create_synthetic_ecg()
    samples[50] = np.nan

    with pytest.raises(
        FeatureExtractionError,
        match="non-finite samples",
    ):
        SciPyRPeakDetector().detect(
            create_signal(samples)
        )


def test_detector_rejects_insufficient_peaks() -> None:
    sampling_rate_hz = 250.0

    time = (
        np.arange(1000, dtype=np.float64)
        / sampling_rate_hz
    )

    samples = 0.05 * np.sin(
        2.0 * np.pi * 0.5 * time
    )

    with pytest.raises(
        FeatureExtractionError,
        match="Insufficient R peaks",
    ):
        SciPyRPeakDetector(
            prominence_threshold=5.0,
        ).detect(
            create_signal(
                samples,
                sampling_rate_hz=sampling_rate_hz,
            )
        )


def test_detector_rejects_sampling_rate_below_filter_limit() -> None:
    samples, _ = create_synthetic_ecg(
        sampling_rate_hz=30.0,
    )

    with pytest.raises(
        UnsupportedSamplingRateError,
        match="too low",
    ):
        SciPyRPeakDetector().detect(
            create_signal(
                samples,
                sampling_rate_hz=30.0,
            )
        )


@pytest.mark.parametrize(
    ("arguments", "expected_message"),
    [
        (
            {"lowcut_hz": 0.0},
            "lowcut_hz must be greater than zero",
        ),
        (
            {
                "lowcut_hz": 20.0,
                "highcut_hz": 10.0,
            },
            "highcut_hz must be greater",
        ),
        (
            {"filter_order": 0},
            "filter_order must be at least one",
        ),
        (
            {"minimum_heart_rate_bpm": 0.0},
            "minimum_heart_rate_bpm must be greater",
        ),
        (
            {
                "minimum_heart_rate_bpm": 100.0,
                "maximum_heart_rate_bpm": 90.0,
            },
            "maximum_heart_rate_bpm must be greater",
        ),
        (
            {"prominence_threshold": 0.0},
            "prominence_threshold must be greater",
        ),
        (
            {"minimum_required_peaks": 1},
            "minimum_required_peaks must be at least two",
        ),
    ],
)
def test_detector_rejects_invalid_configuration(
    arguments: dict[str, float | int],
    expected_message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        SciPyRPeakDetector(**arguments)


def test_detector_output_is_immutable() -> None:
    samples, _ = create_synthetic_ecg()

    result = SciPyRPeakDetector().detect(
        create_signal(samples)
    )

    with pytest.raises(AttributeError):
        result.confidence = 0.0  # type: ignore[misc]