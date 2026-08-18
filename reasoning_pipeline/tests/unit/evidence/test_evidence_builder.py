from __future__ import annotations

import pytest

from reasoning_pipeline.domain.enums.statuses import (
    EvidenceDirection,
    SignalQualityStatus,
)
from reasoning_pipeline.domain.models.feature_set import FeatureSet
from reasoning_pipeline.domain.models.model_prediction import ModelPrediction
from reasoning_pipeline.domain.models.morphology_features import (
    MorphologyFeatures,
)
from reasoning_pipeline.domain.models.r_peak_series import RPeakSeries
from reasoning_pipeline.domain.models.rhythm_features import RhythmFeatures
from reasoning_pipeline.domain.models.signal_quality import SignalQuality
from reasoning_pipeline.evidence.builder import EvidenceBuilder
from reasoning_pipeline.evidence.configuration import (
    EvidenceBuilderConfiguration,
)


def create_prediction(
    *,
    predicted_class: int = 0,
    predicted_label: str = "Normal Sinus Rhythm",
    confidence: float = 0.90,
    probabilities: tuple[float, ...] | None = None,
) -> ModelPrediction:
    if probabilities is None:
        probabilities = (confidence, 1.0 - confidence)

    return ModelPrediction(
        predicted_class=predicted_class,
        predicted_label=predicted_label,
        probabilities=probabilities,
        confidence=confidence,
        checkpoint_path="checkpoints/global_model.pt",
        checkpoint_hash="test-checkpoint-hash",
        model_version="fl-model-v1",
        preprocessing_version="baseline-preprocessing-v1",
    )


def create_feature_set(
    *,
    signal_quality_score: float = 0.90,
    signal_quality_status: SignalQualityStatus = (
        SignalQualityStatus.GOOD
    ),
    signal_quality_warnings: tuple[str, ...] = (),
    r_peak_confidence: float = 0.90,
    heart_rate_mean_bpm: float | None = 75.0,
    irregularity_score: float | None = 0.10,
    mean_qrs_duration_ms: float | None = 90.0,
    abnormal_beat_count: int | None = 0,
    morphology_confidence: float = 0.90,
    mean_pr_interval_ms: float | None = None,
    mean_qt_interval_ms: float | None = None,
    warnings: tuple[str, ...] = (),
) -> FeatureSet:
    return FeatureSet(
        signal_quality=SignalQuality(
            score=signal_quality_score,
            status=signal_quality_status,
            noise_score=0.10,
            valid_sample_ratio=0.99,
            warnings=signal_quality_warnings,
        ),
        r_peaks=RPeakSeries(
            sample_indices=(100, 350, 600, 850),
            timestamps_seconds=(0.4, 1.4, 2.4, 3.4),
            rr_intervals_ms=(1000.0, 1000.0, 1000.0),
            detector_name="scipy-adaptive",
            detector_version="1.0",
            confidence=r_peak_confidence,
        ),
        rhythm=RhythmFeatures(
            heart_rate_mean_bpm=heart_rate_mean_bpm,
            heart_rate_min_bpm=heart_rate_mean_bpm,
            heart_rate_max_bpm=heart_rate_mean_bpm,
            mean_rr_ms=1000.0,
            sdnn_ms=20.0,
            rmssd_ms=18.0,
            pnn50_percent=5.0,
            irregularity_score=irregularity_score,
        ),
        morphology=MorphologyFeatures(
            mean_qrs_duration_ms=mean_qrs_duration_ms,
            mean_pr_interval_ms=mean_pr_interval_ms,
            mean_qt_interval_ms=mean_qt_interval_ms,
            mean_r_amplitude=0.85,
            abnormal_beat_count=abnormal_beat_count,
            morphology_confidence=morphology_confidence,
        ),
        extraction_version="scribe-v2",
        warnings=warnings,
    )


def all_evidence(bundle: object) -> tuple:
    return (
        bundle.supporting_evidence
        + bundle.conflicting_evidence
        + bundle.neutral_evidence
    )


def find_evidence(bundle: object, evidence_id: str):
    matching = [
        item
        for item in all_evidence(bundle)
        if item.evidence_id == evidence_id
    ]

    assert len(matching) == 1
    return matching[0]


def test_builder_returns_evidence_bundle() -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(),
        features=create_feature_set(),
    )

    assert bundle.record_id == "record-001"
    assert bundle.evidence_version == "evidence-builder-v1"
    assert bundle.prediction.predicted_label == "Normal Sinus Rhythm"
    assert bundle.features.extraction_version == "scribe-v2"


def test_builder_rejects_empty_record_id() -> None:
    with pytest.raises(
        ValueError,
        match="record_id cannot be empty",
    ):
        EvidenceBuilder().build(
            record_id=" ",
            prediction=create_prediction(),
            features=create_feature_set(),
        )


def test_model_prediction_is_always_supporting_evidence() -> None:
    prediction = create_prediction(confidence=0.82)

    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=prediction,
        features=create_feature_set(),
    )

    item = find_evidence(bundle, "model.prediction")

    assert item.direction == EvidenceDirection.SUPPORTS
    assert item.measured_value == "Normal Sinus Rhythm"
    assert item.reliability == pytest.approx(0.82)


def test_normal_prediction_with_normal_rate_is_supported() -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(
            predicted_label="Normal Sinus Rhythm",
        ),
        features=create_feature_set(
            heart_rate_mean_bpm=75.0,
        ),
    )

    item = find_evidence(bundle, "rhythm.heart_rate")

    assert item.direction == EvidenceDirection.SUPPORTS
    assert item.measured_value == pytest.approx(75.0)
    assert item.unit == "bpm"


def test_normal_prediction_with_tachycardia_is_conflicting() -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(
            predicted_label="Normal Sinus Rhythm",
        ),
        features=create_feature_set(
            heart_rate_mean_bpm=135.0,
        ),
    )

    item = find_evidence(bundle, "rhythm.heart_rate")

    assert item.direction == EvidenceDirection.CONTRADICTS


def test_supraventricular_prediction_with_tachycardia_is_supported() -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(
            predicted_class=1,
            predicted_label="Supraventricular Ectopic",
        ),
        features=create_feature_set(
            heart_rate_mean_bpm=130.0,
        ),
    )

    item = find_evidence(bundle, "rhythm.heart_rate")

    assert item.direction == EvidenceDirection.SUPPORTS


def test_normal_prediction_with_regular_rhythm_is_supported() -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(
            predicted_label="Normal Sinus Rhythm",
        ),
        features=create_feature_set(
            irregularity_score=0.20,
        ),
    )

    item = find_evidence(bundle, "rhythm.irregularity")

    assert item.direction == EvidenceDirection.SUPPORTS


def test_normal_prediction_with_irregular_rhythm_is_conflicting() -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(
            predicted_label="Normal Sinus Rhythm",
        ),
        features=create_feature_set(
            irregularity_score=0.80,
        ),
    )

    item = find_evidence(bundle, "rhythm.irregularity")

    assert item.direction == EvidenceDirection.CONTRADICTS


@pytest.mark.parametrize(
    "label",
    [
        "Supraventricular Ectopic",
        "Ventricular Arrhythmia",
    ],
)
def test_arrhythmia_prediction_with_irregular_rhythm_is_supported(
    label: str,
) -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(
            predicted_class=1,
            predicted_label=label,
        ),
        features=create_feature_set(
            irregularity_score=0.80,
        ),
    )

    item = find_evidence(bundle, "rhythm.irregularity")

    assert item.direction == EvidenceDirection.SUPPORTS


def test_ventricular_prediction_with_wide_qrs_is_supported() -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(
            predicted_class=2,
            predicted_label="Ventricular Arrhythmia",
        ),
        features=create_feature_set(
            mean_qrs_duration_ms=165.0,
        ),
    )

    item = find_evidence(bundle, "morphology.qrs_duration")

    assert item.direction == EvidenceDirection.SUPPORTS
    assert item.unit == "ms"


def test_ventricular_prediction_with_narrow_qrs_is_conflicting() -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(
            predicted_class=2,
            predicted_label="Ventricular Arrhythmia",
        ),
        features=create_feature_set(
            mean_qrs_duration_ms=90.0,
        ),
    )

    item = find_evidence(bundle, "morphology.qrs_duration")

    assert item.direction == EvidenceDirection.CONTRADICTS


def test_normal_prediction_with_wide_qrs_is_conflicting() -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(),
        features=create_feature_set(
            mean_qrs_duration_ms=150.0,
        ),
    )

    item = find_evidence(bundle, "morphology.qrs_duration")

    assert item.direction == EvidenceDirection.CONTRADICTS


def test_normal_prediction_with_no_abnormal_beats_is_supported() -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(),
        features=create_feature_set(
            abnormal_beat_count=0,
        ),
    )

    item = find_evidence(bundle, "morphology.abnormal_beats")

    assert item.direction == EvidenceDirection.SUPPORTS


def test_normal_prediction_with_abnormal_beats_is_conflicting() -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(),
        features=create_feature_set(
            abnormal_beat_count=4,
        ),
    )

    item = find_evidence(bundle, "morphology.abnormal_beats")

    assert item.direction == EvidenceDirection.CONTRADICTS


@pytest.mark.parametrize(
    "label",
    [
        "Ventricular Arrhythmia",
        "Fusion Beat",
    ],
)
def test_abnormal_prediction_with_abnormal_beats_is_supported(
    label: str,
) -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(
            predicted_class=2,
            predicted_label=label,
        ),
        features=create_feature_set(
            abnormal_beat_count=3,
        ),
    )

    item = find_evidence(bundle, "morphology.abnormal_beats")

    assert item.direction == EvidenceDirection.SUPPORTS


def test_missing_heart_rate_creates_unavailable_evidence() -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(),
        features=create_feature_set(
            heart_rate_mean_bpm=None,
        ),
    )

    item = find_evidence(bundle, "rhythm.heart_rate")

    assert item.direction == EvidenceDirection.UNAVAILABLE
    assert item.measured_value is None
    assert item.reliability == 0.0
    assert "Mean heart rate could not be calculated." in bundle.limitations


def test_missing_irregularity_creates_unavailable_evidence() -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(),
        features=create_feature_set(
            irregularity_score=None,
        ),
    )

    item = find_evidence(bundle, "rhythm.irregularity")

    assert item.direction == EvidenceDirection.UNAVAILABLE
    assert (
        "Rhythm irregularity could not be assessed."
        in bundle.limitations
    )


def test_missing_qrs_duration_creates_unavailable_evidence() -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(),
        features=create_feature_set(
            mean_qrs_duration_ms=None,
        ),
    )

    item = find_evidence(bundle, "morphology.qrs_duration")

    assert item.direction == EvidenceDirection.UNAVAILABLE
    assert (
        "Mean QRS duration could not be estimated."
        in bundle.limitations
    )


def test_missing_abnormal_count_creates_unavailable_evidence() -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(),
        features=create_feature_set(
            abnormal_beat_count=None,
        ),
    )

    item = find_evidence(bundle, "morphology.abnormal_beats")

    assert item.direction == EvidenceDirection.UNAVAILABLE
    assert (
        "Abnormal beat count could not be estimated."
        in bundle.limitations
    )


def test_unavailable_evidence_is_placed_in_neutral_collection() -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(),
        features=create_feature_set(
            heart_rate_mean_bpm=None,
        ),
    )

    unavailable_ids = {
        item.evidence_id
        for item in bundle.neutral_evidence
        if item.direction == EvidenceDirection.UNAVAILABLE
    }

    assert "rhythm.heart_rate" in unavailable_ids


def test_low_model_confidence_adds_limitation() -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(
            confidence=0.55,
            probabilities=(0.55, 0.45),
        ),
        features=create_feature_set(),
    )

    assert (
        "Model confidence was below the configured confidence threshold."
        in bundle.limitations
    )


def test_low_signal_quality_adds_limitation() -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(),
        features=create_feature_set(
            signal_quality_score=0.40,
            signal_quality_status=SignalQualityStatus.POOR,
        ),
    )

    assert (
        "Low signal quality may reduce the reliability "
        "of extracted evidence."
        in bundle.limitations
    )


def test_pr_and_qt_unavailability_are_reported() -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(),
        features=create_feature_set(
            mean_pr_interval_ms=None,
            mean_qt_interval_ms=None,
        ),
    )

    assert "PR interval evidence was unavailable." in bundle.limitations
    assert "QT interval evidence was unavailable." in bundle.limitations


def test_available_pr_and_qt_do_not_add_limitations() -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(),
        features=create_feature_set(
            mean_pr_interval_ms=160.0,
            mean_qt_interval_ms=390.0,
        ),
    )

    assert "PR interval evidence was unavailable." not in bundle.limitations
    assert "QT interval evidence was unavailable." not in bundle.limitations


def test_feature_and_quality_warnings_are_preserved() -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(),
        features=create_feature_set(
            warnings=("Feature extraction warning.",),
            signal_quality_warnings=("Signal quality warning.",),
        ),
    )

    assert "Feature extraction warning." in bundle.limitations
    assert "Signal quality warning." in bundle.limitations


def test_duplicate_limitations_are_removed() -> None:
    repeated_warning = "Repeated warning."

    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(),
        features=create_feature_set(
            warnings=(repeated_warning, repeated_warning),
            signal_quality_warnings=(repeated_warning,),
        ),
    )

    assert bundle.limitations.count(repeated_warning) == 1


def test_rhythm_reliability_uses_quality_and_peak_confidence() -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(),
        features=create_feature_set(
            signal_quality_score=0.80,
            r_peak_confidence=0.60,
        ),
    )

    item = find_evidence(bundle, "rhythm.heart_rate")

    assert item.reliability == pytest.approx(0.70)


def test_morphology_reliability_uses_three_confidence_sources() -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(),
        features=create_feature_set(
            signal_quality_score=0.90,
            r_peak_confidence=0.60,
            morphology_confidence=0.30,
        ),
    )

    item = find_evidence(bundle, "morphology.qrs_duration")

    assert item.reliability == pytest.approx(0.60)


def test_unknown_prediction_with_low_quality_gets_quality_support() -> None:
    bundle = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(
            predicted_class=4,
            predicted_label="Unknown/Unclassifiable",
        ),
        features=create_feature_set(
            signal_quality_score=0.40,
            signal_quality_status=SignalQualityStatus.POOR,
        ),
    )

    item = find_evidence(bundle, "signal.quality")

    assert item.direction == EvidenceDirection.SUPPORTS


def test_custom_configuration_controls_thresholds_and_version() -> None:
    configuration = EvidenceBuilderConfiguration(
        normal_heart_rate_min_bpm=50.0,
        normal_heart_rate_max_bpm=110.0,
        irregularity_threshold=0.70,
        wide_qrs_threshold_ms=140.0,
        evidence_version="evidence-builder-test-v2",
    )

    bundle = EvidenceBuilder(configuration).build(
        record_id="record-001",
        prediction=create_prediction(),
        features=create_feature_set(
            heart_rate_mean_bpm=105.0,
            irregularity_score=0.60,
            mean_qrs_duration_ms=130.0,
        ),
    )

    heart_rate_item = find_evidence(bundle, "rhythm.heart_rate")
    irregularity_item = find_evidence(
        bundle,
        "rhythm.irregularity",
    )
    qrs_item = find_evidence(bundle, "morphology.qrs_duration")

    assert bundle.evidence_version == "evidence-builder-test-v2"
    assert heart_rate_item.direction == EvidenceDirection.SUPPORTS
    assert irregularity_item.direction == EvidenceDirection.SUPPORTS
    assert qrs_item.direction == EvidenceDirection.SUPPORTS


@pytest.mark.parametrize(
    ("arguments", "expected_message"),
    [
        (
            {"normal_heart_rate_min_bpm": 0.0},
            "normal_heart_rate_min_bpm must be greater than zero",
        ),
        (
            {
                "normal_heart_rate_min_bpm": 100.0,
                "normal_heart_rate_max_bpm": 100.0,
            },
            "normal_heart_rate_max_bpm must be greater",
        ),
        (
            {"irregularity_threshold": -0.1},
            "irregularity_threshold must be between",
        ),
        (
            {"irregularity_threshold": 1.1},
            "irregularity_threshold must be between",
        ),
        (
            {"wide_qrs_threshold_ms": 0.0},
            "wide_qrs_threshold_ms must be greater than zero",
        ),
        (
            {"low_model_confidence_threshold": 1.1},
            "low_model_confidence_threshold must be between",
        ),
        (
            {"low_signal_quality_threshold": -0.1},
            "low_signal_quality_threshold must be between",
        ),
        (
            {"evidence_version": " "},
            "evidence_version cannot be empty",
        ),
        (
            {"normal_heart_rate_min_bpm": float("nan")},
            "configuration values must be finite",
        ),
    ],
)
def test_configuration_rejects_invalid_values(
    arguments: dict[str, object],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        EvidenceBuilderConfiguration(**arguments)
