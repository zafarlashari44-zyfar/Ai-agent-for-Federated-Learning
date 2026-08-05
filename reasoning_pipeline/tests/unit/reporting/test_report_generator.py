from __future__ import annotations

import pytest

from reasoning_pipeline.domain.enums.statuses import (
    ConsistencyStatus,
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
from reasoning_pipeline.reasoning.engine import ReasoningEngine
from reasoning_pipeline.reporting.configuration import ReportConfiguration
from reasoning_pipeline.reporting.generator import ReportGenerator


def create_prediction(
    *,
    label: str = "Normal Sinus Rhythm",
    confidence: float = 0.90,
) -> ModelPrediction:
    return ModelPrediction(
        predicted_class=0,
        predicted_label=label,
        probabilities=(confidence, 1.0 - confidence),
        confidence=confidence,
        checkpoint_path="checkpoints/global_model.pt",
        checkpoint_hash="test-checkpoint-hash",
        model_version="fl-model-v1",
        preprocessing_version="baseline-preprocessing-v1",
    )


def create_features(
    *,
    signal_quality_score: float = 0.90,
    heart_rate: float | None = 75.0,
    irregularity: float | None = 0.10,
    qrs_duration: float | None = 90.0,
    abnormal_beats: int | None = 0,
) -> FeatureSet:
    quality_status = SignalQualityStatus.GOOD

    if signal_quality_score < 0.50:
        quality_status = SignalQualityStatus.POOR

    return FeatureSet(
        signal_quality=SignalQuality(
            score=signal_quality_score,
            status=quality_status,
            noise_score=0.10,
            valid_sample_ratio=0.99,
        ),
        r_peaks=RPeakSeries(
            sample_indices=(100, 350, 600, 850),
            timestamps_seconds=(0.4, 1.4, 2.4, 3.4),
            rr_intervals_ms=(1000.0, 1000.0, 1000.0),
            detector_name="scipy-adaptive",
            detector_version="1.0",
            confidence=0.90,
        ),
        rhythm=RhythmFeatures(
            heart_rate_mean_bpm=heart_rate,
            heart_rate_min_bpm=heart_rate,
            heart_rate_max_bpm=heart_rate,
            mean_rr_ms=1000.0,
            sdnn_ms=20.0,
            rmssd_ms=18.0,
            pnn50_percent=5.0,
            irregularity_score=irregularity,
        ),
        morphology=MorphologyFeatures(
            mean_qrs_duration_ms=qrs_duration,
            mean_pr_interval_ms=160.0,
            mean_qt_interval_ms=400.0,
            mean_r_amplitude=0.85,
            abnormal_beat_count=abnormal_beats,
            morphology_confidence=0.90,
        ),
        extraction_version="scribe-v2",
    )


def create_reasoning_result(
    *,
    prediction: ModelPrediction | None = None,
    features: FeatureSet | None = None,
):
    if prediction is None:
        prediction = create_prediction()

    if features is None:
        features = create_features()

    evidence = EvidenceBuilder().build(
        record_id="record-001",
        prediction=prediction,
        features=features,
    )

    return ReasoningEngine().reason(evidence)


def test_generator_returns_structured_clinical_report() -> None:
    reasoning_result = create_reasoning_result()

    report = ReportGenerator().generate(reasoning_result)

    assert report.record_id == "record-001"
    assert report.predicted_label == "Normal Sinus Rhythm"
    assert report.prediction_confidence == pytest.approx(0.90)
    assert report.consistency_status == (
        ConsistencyStatus.STRONGLY_SUPPORTED
    )
    assert report.reasoning_confidence == pytest.approx(
        reasoning_result.reasoning_confidence
    )
    assert report.report_version == "clinical-report-v1"


def test_report_preserves_pipeline_provenance() -> None:
    report = ReportGenerator().generate(
        create_reasoning_result()
    )

    assert report.model_version == "fl-model-v1"
    assert (
        report.preprocessing_version
        == "baseline-preprocessing-v1"
    )
    assert report.evidence_version == "evidence-builder-v1"
    assert report.reasoning_version == "reasoning-engine-v1"


def test_report_contains_supporting_findings() -> None:
    report = ReportGenerator().generate(
        create_reasoning_result()
    )

    assert report.supporting_findings
    assert any(
        "federated model predicted" in finding.lower()
        for finding in report.supporting_findings
    )
    assert any(
        "mean heart rate" in finding.lower()
        for finding in report.supporting_findings
    )


def test_conflicting_findings_are_preserved() -> None:
    reasoning_result = create_reasoning_result(
        features=create_features(
            heart_rate=135.0,
            irregularity=0.80,
            qrs_duration=145.0,
            abnormal_beats=4,
        )
    )

    report = ReportGenerator().generate(reasoning_result)

    assert (
        report.consistency_status
        == ConsistencyStatus.CONFLICTING_EVIDENCE
    )
    assert report.conflicting_findings
    assert any(
        "mean qrs duration" in finding.lower()
        for finding in report.conflicting_findings
    )


def test_summary_contains_both_confidence_values() -> None:
    reasoning_result = create_reasoning_result()

    report = ReportGenerator().generate(reasoning_result)

    assert "90.0% confidence" in report.summary
    assert (
        f"{reasoning_result.reasoning_confidence:.1%}"
        in report.summary
    )


def test_low_signal_quality_produces_acquisition_recommendation() -> None:
    reasoning_result = create_reasoning_result(
        features=create_features(signal_quality_score=0.20)
    )

    report = ReportGenerator().generate(reasoning_result)

    assert (
        report.consistency_status
        == ConsistencyStatus.LOW_SIGNAL_QUALITY
    )
    assert "signal acquisition" in report.recommended_action.lower()
    assert "cleaning the ecg" in report.recommended_action.lower()


def test_conflicting_evidence_produces_manual_review_action() -> None:
    reasoning_result = create_reasoning_result(
        features=create_features(
            heart_rate=140.0,
            irregularity=0.90,
            qrs_duration=150.0,
            abnormal_beats=5,
        )
    )

    report = ReportGenerator().generate(reasoning_result)

    assert "manual review" in report.recommended_action.lower()


def test_custom_configuration_is_used() -> None:
    generator = ReportGenerator(
        configuration=ReportConfiguration(
            report_version="clinical-report-test-v2",
            disclaimer="Test disclaimer.",
        )
    )

    report = generator.generate(create_reasoning_result())

    assert report.report_version == "clinical-report-test-v2"
    assert report.disclaimer == "Test disclaimer."


def test_report_generation_is_deterministic() -> None:
    reasoning_result = create_reasoning_result()
    generator = ReportGenerator()

    first_report = generator.generate(reasoning_result)
    second_report = generator.generate(reasoning_result)

    assert first_report == second_report


def test_report_carries_reasoning_limitations_forward() -> None:
    reasoning_result = create_reasoning_result(
        features=create_features(
            heart_rate=None,
            irregularity=None,
            qrs_duration=None,
            abnormal_beats=None,
        )
    )

    report = ReportGenerator().generate(reasoning_result)

    assert report.limitations
    assert (
        "Mean heart rate could not be calculated."
        in report.limitations
    )
