from __future__ import annotations

import numpy as np
import pytest

from reasoning_pipeline.domain.enums.statuses import SignalQualityStatus
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.domain.models.feature_set import FeatureSet
from reasoning_pipeline.domain.models.morphology_features import (
    MorphologyFeatures,
)
from reasoning_pipeline.domain.models.r_peak_series import RPeakSeries
from reasoning_pipeline.domain.models.rhythm_features import RhythmFeatures
from reasoning_pipeline.domain.models.signal_quality import SignalQuality
from reasoning_pipeline.orchestration.model_input_preparer import (
    ModelInputPreparer,
)


class IdentityCleaner:
    def clean(self, signal: ECGSignal) -> np.ndarray:
        return np.asarray(
            signal.samples,
            dtype=np.float64,
        )


def _build_signal(
    *,
    sample_count: int = 1000,
    sampling_rate_hz: float = 360.0,
) -> ECGSignal:
    samples = np.sin(
        np.linspace(
            0.0,
            20.0,
            sample_count,
        )
    )

    return ECGSignal(
        record_id="record-001",
        samples=tuple(float(value) for value in samples),
        sampling_rate_hz=sampling_rate_hz,
        source="unit-test",
        lead_name="MLII",
    )


def _build_features(
    peak_indices: tuple[int, ...],
) -> FeatureSet:
    r_peaks = RPeakSeries(
        sample_indices=peak_indices,
        timestamps_seconds=tuple(
            index / 360.0
            for index in peak_indices
        ),
        rr_intervals_ms=tuple(
            (
                peak_indices[index]
                - peak_indices[index - 1]
            )
            / 360.0
            * 1000.0
            for index in range(1, len(peak_indices))
        ),
        detector_name="test-detector",
        detector_version="1.0.0",
        confidence=1.0,
    )

    rhythm = RhythmFeatures(
        heart_rate_mean_bpm=None,
        heart_rate_min_bpm=None,
        heart_rate_max_bpm=None,
        mean_rr_ms=None,
        sdnn_ms=None,
        rmssd_ms=None,
        pnn50_percent=None,
        irregularity_score=None,
    )

    morphology = MorphologyFeatures(
        mean_qrs_duration_ms=None,
        mean_pr_interval_ms=None,
        mean_qt_interval_ms=None,
        mean_r_amplitude=None,
        abnormal_beat_count=None,
        morphology_confidence=1.0,
    )

    signal_quality = SignalQuality(
        score=1.0,
        status=SignalQualityStatus.GOOD,
        noise_score=0.0,
        valid_sample_ratio=1.0,
        warnings=(),
    )

    return FeatureSet(
        signal_quality=signal_quality,
        r_peaks=r_peaks,
        rhythm=rhythm,
        morphology=morphology,
        extraction_version="test-version",
        warnings=(),
    )


def test_prepare_all_creates_216_sample_beats() -> None:
    preparer = ModelInputPreparer(
        cleaner=IdentityCleaner(),
    )
    signal = _build_signal()
    features = _build_features((100, 400, 800))

    beats = preparer.prepare_all(
        signal=signal,
        features=features,
    )

    assert len(beats) == 3
    assert all(
        len(beat.samples) == 216
        for beat in beats
    )


def test_each_beat_is_z_normalised() -> None:
    preparer = ModelInputPreparer(
        cleaner=IdentityCleaner(),
    )
    signal = _build_signal()
    features = _build_features((400,))

    beat = preparer.prepare_representative(
        signal=signal,
        features=features,
    )

    values = np.asarray(
        beat.samples,
        dtype=np.float32,
    )

    assert float(values.mean()) == pytest.approx(
        0.0,
        abs=1e-6,
    )
    assert float(values.std()) == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_edge_peaks_are_skipped() -> None:
    preparer = ModelInputPreparer(
        cleaner=IdentityCleaner(),
    )
    signal = _build_signal()
    features = _build_features((20, 400, 950))

    beats = preparer.prepare_all(
        signal=signal,
        features=features,
    )

    assert len(beats) == 1
    assert beats[0].r_peak_sample_index == 400


def test_unsupported_sampling_rate_is_rejected() -> None:
    preparer = ModelInputPreparer(
        cleaner=IdentityCleaner(),
    )
    signal = _build_signal(
        sampling_rate_hz=250.0,
    )
    features = _build_features((400,))

    with pytest.raises(
        ValueError,
        match="supports only 360 Hz",
    ):
        preparer.prepare_all(
            signal=signal,
            features=features,
        )


def test_prepare_matrix_returns_float32_matrix() -> None:
    preparer = ModelInputPreparer(
        cleaner=IdentityCleaner(),
    )
    signal = _build_signal()
    features = _build_features((300, 600))

    matrix = preparer.prepare_matrix(
        signal=signal,
        features=features,
    )

    assert matrix.shape == (2, 216)
    assert matrix.dtype == np.float32


class ReplacingCleaner:
    def clean(self, signal: ECGSignal) -> np.ndarray:
        values = np.arange(
            len(signal.samples),
            dtype=np.float64,
        )

        return np.square(values)


def test_prepare_all_segments_cleaned_signal_not_raw_signal() -> None:
    signal = _build_signal()
    features = _build_features((400,))

    preparer = ModelInputPreparer(
        cleaner=ReplacingCleaner(),
    )

    prepared = preparer.prepare_representative(
        signal=signal,
        features=features,
    )

    cleaned_signal = np.square(
        np.arange(
            len(signal.samples),
            dtype=np.float64,
        )
    )

    expected_window = cleaned_signal[
        400 - 72 : 400 + 144
    ]

    expected = (
        expected_window - expected_window.mean()
    ) / expected_window.std()

    assert np.allclose(
        np.asarray(prepared.samples),
        expected.astype(np.float32),
        atol=1e-6,
    )
