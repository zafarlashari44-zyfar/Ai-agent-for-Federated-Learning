from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import Mock

import pytest

from reasoning_pipeline.domain.enums.statuses import (
    SignalQualityStatus,
)
from reasoning_pipeline.domain.exceptions.pipeline_errors import (
    InsufficientSignalQualityError,
)
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.domain.models.morphology_features import (
    MorphologyFeatures,
)
from reasoning_pipeline.domain.models.r_peak_series import RPeakSeries
from reasoning_pipeline.domain.models.rhythm_features import RhythmFeatures
from reasoning_pipeline.domain.models.signal_quality import SignalQuality
from reasoning_pipeline.scribe_v2.feature_extraction_service import (
    ScribeV2FeatureExtractionService,
)


def create_signal() -> ECGSignal:
    return ECGSignal(
        record_id="record-001",
        samples=(
            0.0,
            0.1,
            1.0,
            0.1,
            0.0,
            0.1,
            1.0,
            0.1,
            0.0,
        ),
        sampling_rate_hz=250.0,
        source="unit-test",
        lead_name="II",
    )


def create_quality(
    *,
    status: SignalQualityStatus = SignalQualityStatus.GOOD,
    warnings: tuple[str, ...] = (),
) -> SignalQuality:
    return SignalQuality(
        score=0.90,
        status=status,
        noise_score=0.10,
        valid_sample_ratio=1.0,
        warnings=warnings,
    )


def create_r_peaks() -> RPeakSeries:
    return RPeakSeries(
        sample_indices=(250, 500, 750),
        timestamps_seconds=(1.0, 2.0, 3.0),
        rr_intervals_ms=(1000.0, 1000.0),
        detector_name="test-detector",
        detector_version="1.0.0",
        confidence=0.90,
    )


def create_rhythm() -> RhythmFeatures:
    return RhythmFeatures(
        heart_rate_mean_bpm=60.0,
        heart_rate_min_bpm=60.0,
        heart_rate_max_bpm=60.0,
        mean_rr_ms=1000.0,
        sdnn_ms=0.0,
        rmssd_ms=0.0,
        pnn50_percent=0.0,
        irregularity_score=0.0,
    )


def create_morphology() -> MorphologyFeatures:
    return MorphologyFeatures(
        mean_qrs_duration_ms=90.0,
        mean_pr_interval_ms=160.0,
        mean_qt_interval_ms=400.0,
        mean_r_amplitude=1.2,
        abnormal_beat_count=0,
        morphology_confidence=0.85,
    )


def build_dependencies() -> tuple[Mock, Mock, Mock]:
    quality_assessor = Mock()
    r_peak_detector = Mock()
    rhythm_extractor = Mock()

    quality_assessor.assess.return_value = create_quality()
    r_peak_detector.detect.return_value = create_r_peaks()
    rhythm_extractor.extract.return_value = create_rhythm()

    return (
        quality_assessor,
        r_peak_detector,
        rhythm_extractor,
    )


def test_service_orchestrates_feature_extraction() -> None:
    (
        quality_assessor,
        r_peak_detector,
        rhythm_extractor,
    ) = build_dependencies()

    signal = create_signal()

    service = ScribeV2FeatureExtractionService(
        quality_assessor=quality_assessor,
        r_peak_detector=r_peak_detector,
        rhythm_extractor=rhythm_extractor,
    )

    result = service.extract(signal)

    quality_assessor.assess.assert_called_once_with(signal)

    r_peak_detector.detect.assert_called_once_with(
        signal,
        signal_quality=quality_assessor.assess.return_value,
    )

    rhythm_extractor.extract.assert_called_once_with(
        r_peak_detector.detect.return_value
    )

    assert result.signal_quality == create_quality()
    assert result.r_peaks == create_r_peaks()
    assert result.rhythm == create_rhythm()
    assert result.extraction_version == "scribe-v2.0.0"


def test_service_returns_explicit_empty_morphology() -> None:
    (
        quality_assessor,
        r_peak_detector,
        rhythm_extractor,
    ) = build_dependencies()

    service = ScribeV2FeatureExtractionService(
        quality_assessor=quality_assessor,
        r_peak_detector=r_peak_detector,
        rhythm_extractor=rhythm_extractor,
    )

    result = service.extract(create_signal())

    assert result.morphology == MorphologyFeatures(
        mean_qrs_duration_ms=None,
        mean_pr_interval_ms=None,
        mean_qt_interval_ms=None,
        mean_r_amplitude=None,
        abnormal_beat_count=None,
        morphology_confidence=0.0,
    )

    assert (
        ScribeV2FeatureExtractionService
        .MORPHOLOGY_UNAVAILABLE_WARNING
        in result.warnings
    )


def test_service_uses_configured_morphology_extractor() -> None:
    (
        quality_assessor,
        r_peak_detector,
        rhythm_extractor,
    ) = build_dependencies()

    morphology_extractor = Mock()
    morphology_extractor.extract.return_value = create_morphology()

    signal = create_signal()

    service = ScribeV2FeatureExtractionService(
        quality_assessor=quality_assessor,
        r_peak_detector=r_peak_detector,
        rhythm_extractor=rhythm_extractor,
        morphology_extractor=morphology_extractor,
    )

    result = service.extract(signal)

    morphology_extractor.extract.assert_called_once_with(
        signal,
        create_r_peaks(),
    )

    assert result.morphology == create_morphology()

    assert (
        ScribeV2FeatureExtractionService
        .MORPHOLOGY_UNAVAILABLE_WARNING
        not in result.warnings
    )


def test_service_rejects_unusable_signal() -> None:
    (
        quality_assessor,
        r_peak_detector,
        rhythm_extractor,
    ) = build_dependencies()

    quality_assessor.assess.return_value = create_quality(
        status=SignalQualityStatus.UNUSABLE,
        warnings=("Signal is predominantly flat.",),
    )

    service = ScribeV2FeatureExtractionService(
        quality_assessor=quality_assessor,
        r_peak_detector=r_peak_detector,
        rhythm_extractor=rhythm_extractor,
    )

    with pytest.raises(
        InsufficientSignalQualityError,
        match="ECG signal is unusable",
    ):
        service.extract(create_signal())

    r_peak_detector.detect.assert_not_called()
    rhythm_extractor.extract.assert_not_called()


@pytest.mark.parametrize(
    "status",
    [
        SignalQualityStatus.GOOD,
        SignalQualityStatus.ACCEPTABLE,
        SignalQualityStatus.POOR,
    ],
)
def test_service_allows_non_unusable_quality_statuses(
    status: SignalQualityStatus,
) -> None:
    (
        quality_assessor,
        r_peak_detector,
        rhythm_extractor,
    ) = build_dependencies()

    quality_assessor.assess.return_value = create_quality(
        status=status,
    )

    service = ScribeV2FeatureExtractionService(
        quality_assessor=quality_assessor,
        r_peak_detector=r_peak_detector,
        rhythm_extractor=rhythm_extractor,
    )

    result = service.extract(create_signal())

    assert result.signal_quality.status is status
    r_peak_detector.detect.assert_called_once()


def test_service_preserves_quality_warnings() -> None:
    (
        quality_assessor,
        r_peak_detector,
        rhythm_extractor,
    ) = build_dependencies()

    quality_warning = (
        "High-frequency noise may reduce feature reliability."
    )

    quality_assessor.assess.return_value = create_quality(
        status=SignalQualityStatus.ACCEPTABLE,
        warnings=(quality_warning,),
    )

    service = ScribeV2FeatureExtractionService(
        quality_assessor=quality_assessor,
        r_peak_detector=r_peak_detector,
        rhythm_extractor=rhythm_extractor,
    )

    result = service.extract(create_signal())

    assert result.warnings[0] == quality_warning
    assert (
        ScribeV2FeatureExtractionService
        .MORPHOLOGY_UNAVAILABLE_WARNING
        in result.warnings
    )


def test_warning_merging_removes_duplicates() -> None:
    warning = "Repeated warning."

    result = (
        ScribeV2FeatureExtractionService._merge_warnings(
            (warning, warning),
            ("", "  ", warning),
        )
    )

    assert result == (warning,)


def test_custom_extraction_version_is_used() -> None:
    (
        quality_assessor,
        r_peak_detector,
        rhythm_extractor,
    ) = build_dependencies()

    service = ScribeV2FeatureExtractionService(
        quality_assessor=quality_assessor,
        r_peak_detector=r_peak_detector,
        rhythm_extractor=rhythm_extractor,
        extraction_version="research-ablation-1",
    )

    result = service.extract(create_signal())

    assert result.extraction_version == "research-ablation-1"


@pytest.mark.parametrize(
    "version",
    [
        "",
        " ",
        "   ",
    ],
)
def test_service_rejects_empty_extraction_version(
    version: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="extraction_version cannot be empty",
    ):
        ScribeV2FeatureExtractionService(
            extraction_version=version
        )


def test_result_is_immutable() -> None:
    (
        quality_assessor,
        r_peak_detector,
        rhythm_extractor,
    ) = build_dependencies()

    service = ScribeV2FeatureExtractionService(
        quality_assessor=quality_assessor,
        r_peak_detector=r_peak_detector,
        rhythm_extractor=rhythm_extractor,
    )

    result = service.extract(create_signal())

    with pytest.raises(FrozenInstanceError):
        result.extraction_version = "changed"  # type: ignore[misc]


def test_unusable_error_includes_quality_warning() -> None:
    (
        quality_assessor,
        r_peak_detector,
        rhythm_extractor,
    ) = build_dependencies()

    quality_assessor.assess.return_value = create_quality(
        status=SignalQualityStatus.UNUSABLE,
        warnings=("Signal contains no finite samples.",),
    )

    service = ScribeV2FeatureExtractionService(
        quality_assessor=quality_assessor,
        r_peak_detector=r_peak_detector,
        rhythm_extractor=rhythm_extractor,
    )

    with pytest.raises(
        InsufficientSignalQualityError,
        match="Signal contains no finite samples",
    ):
        service.extract(create_signal())