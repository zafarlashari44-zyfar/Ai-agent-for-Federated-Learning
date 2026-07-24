from __future__ import annotations

import pytest

from reasoning_pipeline.domain.enums.statuses import SignalQualityStatus
from reasoning_pipeline.domain.models import (
    MorphologyFeatures,
    RPeakSeries,
    RhythmFeatures,
    SignalQuality,
)


def test_signal_quality_accepts_valid_score() -> None:
    quality = SignalQuality(
        score=0.92,
        status=SignalQualityStatus.GOOD,
        noise_score=0.08,
        valid_sample_ratio=0.99,
    )

    assert quality.status is SignalQualityStatus.GOOD
    assert quality.score == pytest.approx(0.92)


def test_signal_quality_rejects_invalid_score() -> None:
    with pytest.raises(
        ValueError,
        match="score must be between zero and one",
    ):
        SignalQuality(
            score=1.2,
            status=SignalQualityStatus.GOOD,
        )


def test_r_peak_series_reports_peak_count() -> None:
    peaks = RPeakSeries(
        sample_indices=(100, 200, 300),
        timestamps_seconds=(1.0, 2.0, 3.0),
        rr_intervals_ms=(1000.0, 1000.0),
        detector_name="test_detector",
        detector_version="1.0",
        confidence=0.95,
    )

    assert peaks.peak_count == 3


def test_r_peak_series_rejects_mismatched_positions() -> None:
    with pytest.raises(
        ValueError,
        match="sample_indices and timestamps_seconds must match",
    ):
        RPeakSeries(
            sample_indices=(100, 200),
            timestamps_seconds=(1.0,),
            rr_intervals_ms=(1000.0,),
            detector_name="test_detector",
            detector_version="1.0",
            confidence=0.95,
        )


def test_rhythm_features_reject_invalid_irregularity_score() -> None:
    with pytest.raises(
        ValueError,
        match="irregularity_score must be between zero and one",
    ):
        RhythmFeatures(
            heart_rate_mean_bpm=75.0,
            heart_rate_min_bpm=60.0,
            heart_rate_max_bpm=90.0,
            mean_rr_ms=800.0,
            sdnn_ms=40.0,
            rmssd_ms=35.0,
            pnn50_percent=10.0,
            irregularity_score=1.3,
        )


def test_morphology_rejects_negative_abnormal_beat_count() -> None:
    with pytest.raises(
        ValueError,
        match="abnormal_beat_count cannot be negative",
    ):
        MorphologyFeatures(
            mean_qrs_duration_ms=90.0,
            mean_pr_interval_ms=160.0,
            mean_qt_interval_ms=400.0,
            mean_r_amplitude=0.8,
            abnormal_beat_count=-1,
            morphology_confidence=0.9,
        )
