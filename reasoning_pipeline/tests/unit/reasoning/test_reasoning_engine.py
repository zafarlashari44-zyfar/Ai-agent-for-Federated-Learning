from __future__ import annotations

import pytest

from reasoning_pipeline.domain.enums.statuses import (
    ConsistencyStatus,
    EvidenceDirection,
    SignalQualityStatus,
)
from reasoning_pipeline.domain.models.evidence_bundle import EvidenceBundle
from reasoning_pipeline.domain.models.evidence_item import EvidenceItem
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


def create_prediction(
    *,
    predicted_label: str = "Normal Sinus Rhythm",
    confidence: float = 0.90,
) -> ModelPrediction:
    return ModelPrediction(
        predicted_class=0,
        predicted_label=predicted_label,
        probabilities=(confidence, 1.0 - confidence),
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
) -> FeatureSet:
    return FeatureSet(
        signal_quality=SignalQuality(
            score=signal_quality_score,
            status=signal_quality_status,
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
            heart_rate_mean_bpm=75.0,
            heart_rate_min_bpm=75.0,
            heart_rate_max_bpm=75.0,
            mean_rr_ms=1000.0,
            sdnn_ms=20.0,
            rmssd_ms=18.0,
            pnn50_percent=5.0,
            irregularity_score=0.10,
        ),
        morphology=MorphologyFeatures(
            mean_qrs_duration_ms=90.0,
            mean_pr_interval_ms=160.0,
            mean_qt_interval_ms=400.0,
            mean_r_amplitude=0.85,
            abnormal_beat_count=0,
            morphology_confidence=0.90,
        ),
        extraction_version="scribe-v2",
    )


def create_evidence_item(
    *,
    evidence_id: str,
    direction: EvidenceDirection,
    reliability: float,
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        feature_name=evidence_id,
        measured_value=1.0,
        unit=None,
        interpretation=f"Evidence item {evidence_id}.",
        direction=direction,
        reliability=reliability,
        source_reference=f"tests.{evidence_id}",
    )


def create_bundle(
    *,
    supporting: tuple[EvidenceItem, ...] = (),
    conflicting: tuple[EvidenceItem, ...] = (),
    neutral: tuple[EvidenceItem, ...] = (),
    signal_quality_score: float = 0.90,
    limitations: tuple[str, ...] = (),
    prediction_confidence: float = 0.90,
) -> EvidenceBundle:
    status = SignalQualityStatus.GOOD

    if signal_quality_score < 0.50:
        status = SignalQualityStatus.POOR

    return EvidenceBundle(
        record_id="record-001",
        prediction=create_prediction(
            confidence=prediction_confidence,
        ),
        features=create_feature_set(
            signal_quality_score=signal_quality_score,
            signal_quality_status=status,
        ),
        supporting_evidence=supporting,
        conflicting_evidence=conflicting,
        neutral_evidence=neutral,
        limitations=limitations,
        evidence_version="evidence-builder-v1",
    )


def test_reasoning_engine_returns_strongly_supported_result() -> None:
    evidence = EvidenceBuilder().build(
        record_id="record-001",
        prediction=create_prediction(),
        features=create_feature_set(),
    )

    result = ReasoningEngine().reason(evidence)

    assert (
        result.consistency_status
        == ConsistencyStatus.STRONGLY_SUPPORTED
    )
    assert result.reasoning_confidence > 0.70
    assert "strongly supports" in result.conclusion
    assert result.evidence is evidence
    assert result.reasoning_version == "reasoning-engine-v1"


def test_reasoning_engine_returns_partially_supported_result() -> None:
    evidence = create_bundle(
        supporting=(
            create_evidence_item(
                evidence_id="support-one",
                direction=EvidenceDirection.SUPPORTS,
                reliability=0.70,
            ),
        ),
        conflicting=(
            create_evidence_item(
                evidence_id="conflict-one",
                direction=EvidenceDirection.CONTRADICTS,
                reliability=0.30,
            ),
        ),
    )

    result = ReasoningEngine().reason(evidence)

    assert (
        result.consistency_status
        == ConsistencyStatus.PARTIALLY_SUPPORTED
    )
    assert "partially supports" in result.conclusion


def test_reasoning_engine_detects_conflicting_evidence() -> None:
    evidence = create_bundle(
        supporting=(
            create_evidence_item(
                evidence_id="support-one",
                direction=EvidenceDirection.SUPPORTS,
                reliability=0.60,
            ),
        ),
        conflicting=(
            create_evidence_item(
                evidence_id="conflict-one",
                direction=EvidenceDirection.CONTRADICTS,
                reliability=0.40,
            ),
        ),
    )

    result = ReasoningEngine().reason(evidence)

    assert (
        result.consistency_status
        == ConsistencyStatus.CONFLICTING_EVIDENCE
    )
    assert "meaningful conflicts" in result.conclusion
    assert any(
        "Supporting and conflicting evidence were both present."
        == limitation
        for limitation in result.limitations
    )


def test_reasoning_engine_detects_insufficient_evidence() -> None:
    evidence = create_bundle(
        neutral=(
            create_evidence_item(
                evidence_id="neutral-one",
                direction=EvidenceDirection.NEUTRAL,
                reliability=0.90,
            ),
        ),
    )

    result = ReasoningEngine().reason(evidence)

    assert (
        result.consistency_status
        == ConsistencyStatus.INSUFFICIENT_EVIDENCE
    )
    assert result.reasoning_confidence == pytest.approx(0.0)
    assert "insufficient reliable evidence" in result.conclusion


def test_low_signal_quality_has_priority_over_support() -> None:
    evidence = create_bundle(
        supporting=(
            create_evidence_item(
                evidence_id="support-one",
                direction=EvidenceDirection.SUPPORTS,
                reliability=0.95,
            ),
            create_evidence_item(
                evidence_id="support-two",
                direction=EvidenceDirection.SUPPORTS,
                reliability=0.95,
            ),
        ),
        signal_quality_score=0.20,
    )

    result = ReasoningEngine().reason(evidence)

    assert (
        result.consistency_status
        == ConsistencyStatus.LOW_SIGNAL_QUALITY
    )
    assert "low signal quality" in result.conclusion.lower()
    assert result.reasoning_confidence < 0.40


def test_conflicting_evidence_reduces_reasoning_confidence() -> None:
    supporting = (
        create_evidence_item(
            evidence_id="support-one",
            direction=EvidenceDirection.SUPPORTS,
            reliability=0.90,
        ),
        create_evidence_item(
            evidence_id="support-two",
            direction=EvidenceDirection.SUPPORTS,
            reliability=0.90,
        ),
    )

    evidence_without_conflict = create_bundle(
        supporting=supporting,
        prediction_confidence=0.95,
    )

    evidence_with_conflict = create_bundle(
        supporting=supporting,
        conflicting=(
            create_evidence_item(
                evidence_id="conflict-one",
                direction=EvidenceDirection.CONTRADICTS,
                reliability=0.80,
            ),
        ),
        prediction_confidence=0.95,
    )

    engine = ReasoningEngine()

    result_without_conflict = engine.reason(
        evidence_without_conflict
    )
    result_with_conflict = engine.reason(evidence_with_conflict)

    assert (
        result_with_conflict.reasoning_confidence
        < result_without_conflict.reasoning_confidence
    )


def test_reasoning_engine_deduplicates_limitations() -> None:
    repeated_limitation = "A repeated pipeline limitation."

    evidence = create_bundle(
        supporting=(
            create_evidence_item(
                evidence_id="support-one",
                direction=EvidenceDirection.SUPPORTS,
                reliability=0.90,
            ),
            create_evidence_item(
                evidence_id="support-two",
                direction=EvidenceDirection.SUPPORTS,
                reliability=0.90,
            ),
        ),
        limitations=(
            repeated_limitation,
            repeated_limitation,
            repeated_limitation,
        ),
    )

    result = ReasoningEngine().reason(evidence)

    assert result.limitations.count(repeated_limitation) == 1


def test_reasoning_engine_creates_deterministic_rule_trace() -> None:
    evidence = create_bundle(
        supporting=(
            create_evidence_item(
                evidence_id="support-one",
                direction=EvidenceDirection.SUPPORTS,
                reliability=0.90,
            ),
            create_evidence_item(
                evidence_id="support-two",
                direction=EvidenceDirection.SUPPORTS,
                reliability=0.80,
            ),
        ),
    )

    engine = ReasoningEngine()

    first_result = engine.reason(evidence)
    second_result = engine.reason(evidence)

    assert first_result.rule_trace == second_result.rule_trace
    assert len(first_result.rule_trace) == 6
    assert any(
        "supporting evidence weight" in step.lower()
        for step in first_result.rule_trace
    )
    assert any(
        "strongly_supported" in step
        for step in first_result.rule_trace
    )


def test_low_average_reliability_is_insufficient() -> None:
    evidence = create_bundle(
        supporting=(
            create_evidence_item(
                evidence_id="support-one",
                direction=EvidenceDirection.SUPPORTS,
                reliability=0.20,
            ),
            create_evidence_item(
                evidence_id="support-two",
                direction=EvidenceDirection.SUPPORTS,
                reliability=0.30,
            ),
        ),
    )

    result = ReasoningEngine().reason(evidence)

    assert (
        result.consistency_status
        == ConsistencyStatus.INSUFFICIENT_EVIDENCE
    )
    assert any(
        "Average evidence reliability was below"
        in limitation
        for limitation in result.limitations
    )


def test_single_reliable_support_is_only_partial_support() -> None:
    evidence = create_bundle(
        supporting=(
            create_evidence_item(
                evidence_id="support-one",
                direction=EvidenceDirection.SUPPORTS,
                reliability=0.90,
            ),
        ),
    )

    result = ReasoningEngine().reason(evidence)

    assert (
        result.consistency_status
        == ConsistencyStatus.PARTIALLY_SUPPORTED
    )


def test_reasoning_result_preserves_existing_limitations() -> None:
    evidence = create_bundle(
        supporting=(
            create_evidence_item(
                evidence_id="support-one",
                direction=EvidenceDirection.SUPPORTS,
                reliability=0.90,
            ),
            create_evidence_item(
                evidence_id="support-two",
                direction=EvidenceDirection.SUPPORTS,
                reliability=0.90,
            ),
        ),
        limitations=("PR interval evidence was unavailable.",),
    )

    result = ReasoningEngine().reason(evidence)

    assert (
        "PR interval evidence was unavailable."
        in result.limitations
    )
