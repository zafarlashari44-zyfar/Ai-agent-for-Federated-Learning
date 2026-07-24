from __future__ import annotations

import math

import numpy as np
import pytest

from reasoning_pipeline.domain.exceptions.pipeline_errors import (
    FeatureExtractionError,
    UnsupportedSamplingRateError,
)
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.domain.models.r_peak_series import RPeakSeries
from reasoning_pipeline.scribe_v2.morphology_feature_extractor import (
    MorphologyFeatureExtractor,
)


def create_synthetic_ecg(
    *,
    sampling_rate_hz: float = 250.0,
    duration_seconds: float = 10.0,
    beat_interval_seconds: float = 1.0,
    qrs_width_ms: float = 80.0,
    amplitude: float = 1.0,
) -> tuple[ECGSignal, RPeakSeries]:
    sample_count = int(
        sampling_rate_hz * duration_seconds
    )

    samples = np.zeros(
        sample_count,
        dtype=np.float64,
    )

    peak_indices: list[int] = []

    qrs_sigma_samples = (
        qrs_width_ms
        / 1000.0
        * sampling_rate_hz
        / 6.0
    )

    beat_times = np.arange(
        1.0,
        duration_seconds - 0.5,
        beat_interval_seconds,
    )

    sample_axis = np.arange(
        sample_count,
        dtype=np.float64,
    )

    for beat_time in beat_times:
        peak_index = int(
            round(beat_time * sampling_rate_hz)
        )

        peak_indices.append(peak_index)

        samples += amplitude * np.exp(
            -0.5
            * (
                (sample_axis - peak_index)
                / qrs_sigma_samples
            )
            ** 2
        )

    timestamps = tuple(
        index / sampling_rate_hz
        for index in peak_indices
    )

    rr_intervals = tuple(
        (
            peak_indices[index]
            - peak_indices[index - 1]
        )
        / sampling_rate_hz
        * 1000.0
        for index in range(1, len(peak_indices))
    )

    signal = ECGSignal(
        record_id="synthetic-morphology",
        samples=tuple(float(value) for value in samples),
        sampling_rate_hz=sampling_rate_hz,
        source="unit-test",
        lead_name="II",
    )

    r_peaks = RPeakSeries(
        sample_indices=tuple(peak_indices),
        timestamps_seconds=timestamps,
        rr_intervals_ms=rr_intervals,
        detector_name="synthetic-detector",
        detector_version="1.0.0",
        confidence=0.95,
    )

    return signal, r_peaks


def test_extractor_returns_morphology_features() -> None:
    signal, r_peaks = create_synthetic_ecg()

    result = MorphologyFeatureExtractor().extract(
        signal,
        r_peaks,
    )

    assert result.mean_qrs_duration_ms is not None
    assert result.mean_r_amplitude is not None
    assert result.abnormal_beat_count is not None
    assert 0.0 <= result.morphology_confidence <= 1.0


def test_extractor_leaves_pr_and_qt_unavailable() -> None:
    signal, r_peaks = create_synthetic_ecg()

    result = MorphologyFeatureExtractor().extract(
        signal,
        r_peaks,
    )

    assert result.mean_pr_interval_ms is None
    assert result.mean_qt_interval_ms is None


def test_extractor_estimates_positive_r_amplitude() -> None:
    signal, r_peaks = create_synthetic_ecg(
        amplitude=1.5,
    )

    result = MorphologyFeatureExtractor().extract(
        signal,
        r_peaks,
    )

    assert result.mean_r_amplitude is not None
    assert result.mean_r_amplitude > 0.0


def test_extractor_supports_inverted_r_peaks() -> None:
    signal, r_peaks = create_synthetic_ecg(
        amplitude=-1.0,
    )

    result = MorphologyFeatureExtractor().extract(
        signal,
        r_peaks,
    )

    assert result.mean_r_amplitude is not None
    assert result.mean_r_amplitude < 0.0


def test_regular_beats_produce_high_confidence() -> None:
    signal, r_peaks = create_synthetic_ecg()

    result = MorphologyFeatureExtractor().extract(
        signal,
        r_peaks,
    )

    assert result.morphology_confidence >= 0.70


def test_extractor_detects_abnormally_wide_qrs_beats() -> None:
    signal, r_peaks = create_synthetic_ecg(
        qrs_width_ms=220.0,
    )

    extractor = MorphologyFeatureExtractor(
        maximum_qrs_duration_ms=130.0,
    )

    result = extractor.extract(
        signal,
        r_peaks,
    )

    assert result.mean_qrs_duration_ms is not None
    assert result.mean_qrs_duration_ms > 130.0
    assert result.abnormal_beat_count is not None
    assert result.abnormal_beat_count > 0


def test_extractor_rejects_insufficient_peaks() -> None:
    signal, r_peaks = create_synthetic_ecg()

    shortened_peaks = RPeakSeries(
        sample_indices=(r_peaks.sample_indices[0],),
        timestamps_seconds=(r_peaks.timestamps_seconds[0],),
        rr_intervals_ms=(),
        detector_name="synthetic-detector",
        detector_version="1.0.0",
        confidence=0.9,
    )

    with pytest.raises(
        FeatureExtractionError,
        match="Insufficient R peaks",
    ):
        MorphologyFeatureExtractor().extract(
            signal,
            shortened_peaks,
        )


def test_extractor_rejects_peak_outside_signal() -> None:
    signal, r_peaks = create_synthetic_ecg()

    invalid_peaks = RPeakSeries(
        sample_indices=(
            r_peaks.sample_indices[0],
            len(signal.samples) + 10,
        ),
        timestamps_seconds=(1.0, 20.0),
        rr_intervals_ms=(19000.0,),
        detector_name="synthetic-detector",
        detector_version="1.0.0",
        confidence=0.9,
    )

    with pytest.raises(
        FeatureExtractionError,
        match="exceeds the ECG signal length",
    ):
        MorphologyFeatureExtractor().extract(
            signal,
            invalid_peaks,
        )


def test_extractor_rejects_low_sampling_rate() -> None:
    signal, r_peaks = create_synthetic_ecg(
        sampling_rate_hz=60.0,
    )

    with pytest.raises(
        UnsupportedSamplingRateError,
        match="(?i)sampling rate is too low",
    ):
        MorphologyFeatureExtractor().extract(
            signal,
            r_peaks,
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {"lowcut_hz": 0.0},
            "lowcut_hz must be greater than zero",
        ),
        (
            {
                "lowcut_hz": 10.0,
                "highcut_hz": 5.0,
            },
            "highcut_hz must be greater",
        ),
        (
            {"filter_order": 0},
            "filter_order must be at least one",
        ),
        (
            {"boundary_threshold_ratio": 0.0},
            "boundary_threshold_ratio must be between",
        ),
        (
            {"boundary_threshold_ratio": 1.0},
            "boundary_threshold_ratio must be between",
        ),
        (
            {
                "minimum_qrs_duration_ms": 150.0,
                "maximum_qrs_duration_ms": 100.0,
            },
            "maximum_qrs_duration_ms must be greater",
        ),
        (
            {"minimum_required_beats": 1},
            "minimum_required_beats must be at least two",
        ),
        (
            {"pre_r_window_ms": math.inf},
            "configuration must be finite",
        ),
    ],
)
def test_extractor_rejects_invalid_configuration(
    arguments: dict[str, float | int],
    message: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=message,
    ):
        MorphologyFeatureExtractor(**arguments)