from __future__ import annotations

import numpy as np
import pytest

from reasoning_pipeline.domain.enums.statuses import SignalQualityStatus
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.scribe_v2.quality_assessor import (
    ECGSignalQualityAssessor,
)


def create_ecg_signal(
    samples: np.ndarray,
    *,
    sampling_rate_hz: float = 250.0,
) -> ECGSignal:
    return ECGSignal(
        record_id="test-record",
        samples=tuple(float(value) for value in samples),
        sampling_rate_hz=sampling_rate_hz,
        source="synthetic",
        lead_name="Lead II",
    )


def create_clean_ecg(
    *,
    sampling_rate_hz: float = 250.0,
    duration_seconds: float = 10.0,
) -> np.ndarray:
    sample_count = int(sampling_rate_hz * duration_seconds)

    time = np.arange(sample_count, dtype=np.float64) / sampling_rate_hz

    base_wave = 0.12 * np.sin(2.0 * np.pi * 1.2 * time)
    qrs_like_component = 0.70 * (
        np.sin(2.0 * np.pi * 1.2 * time) ** 15
    )

    return base_wave + qrs_like_component


def test_assessor_returns_signal_quality_for_clean_signal() -> None:
    signal = create_ecg_signal(create_clean_ecg())

    result = ECGSignalQualityAssessor().assess(signal)

    assert 0.0 <= result.score <= 1.0
    assert result.status in {
        SignalQualityStatus.GOOD,
        SignalQualityStatus.ACCEPTABLE,
    }
    assert result.noise_score is not None
    assert 0.0 <= result.noise_score <= 1.0
    assert result.valid_sample_ratio == pytest.approx(1.0)


def test_assessor_marks_constant_signal_as_unusable() -> None:
    samples = np.ones(2500, dtype=np.float64)
    signal = create_ecg_signal(samples)

    result = ECGSignalQualityAssessor().assess(signal)

    assert result.status is SignalQualityStatus.UNUSABLE
    assert result.score < 0.35
    assert any(
        "flat" in warning.lower()
        or "amplitude" in warning.lower()
        for warning in result.warnings
    )


def test_assessor_detects_flatline_segment() -> None:
    samples = create_clean_ecg()
    samples[500:1800] = 0.0

    signal = create_ecg_signal(samples)

    result = ECGSignalQualityAssessor().assess(signal)

    assert result.status in {
        SignalQualityStatus.POOR,
        SignalQualityStatus.UNUSABLE,
    }
    assert any(
        "flatline" in warning.lower()
        for warning in result.warnings
    )


def test_assessor_detects_high_frequency_noise() -> None:
    clean_signal = create_clean_ecg()

    alternating_noise = np.where(
        np.arange(clean_signal.size) % 2 == 0,
        1.0,
        -1.0,
    )

    noisy_signal = clean_signal + 0.8 * alternating_noise

    result = ECGSignalQualityAssessor().assess(
        create_ecg_signal(noisy_signal)
    )

    assert result.noise_score is not None
    assert result.noise_score >= 0.70
    assert any(
        "noise" in warning.lower()
        for warning in result.warnings
    )


def test_assessor_detects_repeated_extreme_values() -> None:
    samples = create_clean_ecg()

    samples[:250] = np.max(samples)
    samples[250:500] = np.min(samples)

    result = ECGSignalQualityAssessor().assess(
        create_ecg_signal(samples)
    )

    assert any(
        "clipping" in warning.lower()
        for warning in result.warnings
    )


def test_assessor_detects_baseline_instability() -> None:
    sampling_rate_hz = 250.0
    samples = create_clean_ecg(
        sampling_rate_hz=sampling_rate_hz,
    )

    time = (
        np.arange(samples.size, dtype=np.float64)
        / sampling_rate_hz
    )

    baseline_drift = 0.9 * np.sin(
        2.0 * np.pi * 0.12 * time
    )

    result = ECGSignalQualityAssessor().assess(
        create_ecg_signal(
            samples + baseline_drift,
            sampling_rate_hz=sampling_rate_hz,
        )
    )

    assert any(
        "baseline" in warning.lower()
        for warning in result.warnings
    )


def test_assessor_reports_invalid_sample_ratio() -> None:
    samples = create_clean_ecg()
    samples[:100] = np.nan

    signal = create_ecg_signal(samples)

    result = ECGSignalQualityAssessor().assess(signal)

    expected_ratio = (samples.size - 100) / samples.size

    assert result.valid_sample_ratio == pytest.approx(
        expected_ratio
    )
    assert any(
        "non-finite" in warning.lower()
        for warning in result.warnings
    )


def test_assessor_marks_signal_with_too_many_invalid_samples_unusable(
) -> None:
    samples = create_clean_ecg()
    samples[:1000] = np.nan

    result = ECGSignalQualityAssessor().assess(
        create_ecg_signal(samples)
    )

    assert result.status is SignalQualityStatus.UNUSABLE
    assert result.valid_sample_ratio is not None
    assert result.valid_sample_ratio < 0.80


def test_assessor_handles_all_non_finite_samples() -> None:
    samples = np.full(1000, np.nan, dtype=np.float64)

    result = ECGSignalQualityAssessor().assess(
        create_ecg_signal(samples)
    )

    assert result.score == pytest.approx(0.0)
    assert result.status is SignalQualityStatus.UNUSABLE
    assert result.noise_score == pytest.approx(1.0)
    assert result.valid_sample_ratio == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("parameter_name", "parameter_value"),
    [
        ("good_threshold", -0.1),
        ("acceptable_threshold", 1.1),
        ("poor_threshold", -1.0),
        ("minimum_valid_sample_ratio", 2.0),
        ("maximum_flatline_ratio", -0.5),
    ],
)
def test_assessor_rejects_thresholds_outside_unit_interval(
    parameter_name: str,
    parameter_value: float,
) -> None:
    arguments = {parameter_name: parameter_value}

    with pytest.raises(
        ValueError,
        match="must be between zero and one",
    ):
        ECGSignalQualityAssessor(**arguments)


def test_assessor_rejects_invalid_threshold_order() -> None:
    with pytest.raises(
        ValueError,
        match="good > acceptable > poor",
    ):
        ECGSignalQualityAssessor(
            good_threshold=0.50,
            acceptable_threshold=0.70,
            poor_threshold=0.30,
        )


def test_quality_result_is_immutable() -> None:
    result = ECGSignalQualityAssessor().assess(
        create_ecg_signal(create_clean_ecg())
    )

    with pytest.raises(AttributeError):
        result.score = 0.0  # type: ignore[misc]