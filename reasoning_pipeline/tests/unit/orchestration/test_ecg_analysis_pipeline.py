from __future__ import annotations

from dataclasses import dataclass

from reasoning_pipeline.api.schemas.analyse import AnalysisResponse
from reasoning_pipeline.domain.enums.statuses import (
    AnalysisScope,
    ConsistencyStatus,
    SignalQualityStatus,
)
from reasoning_pipeline.domain.models.clinical_report import (
    ClinicalReport,
)
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.domain.models.evidence_bundle import (
    EvidenceBundle,
)
from reasoning_pipeline.domain.models.feature_set import FeatureSet
from reasoning_pipeline.domain.models.model_prediction import (
    ModelPrediction,
)
from reasoning_pipeline.domain.models.morphology_features import (
    MorphologyFeatures,
)
from reasoning_pipeline.domain.models.narrative_result import (
    NarrativeResult,
)
from reasoning_pipeline.domain.models.r_peak_series import RPeakSeries
from reasoning_pipeline.domain.models.reasoning_result import (
    ReasoningResult,
)
from reasoning_pipeline.domain.models.rhythm_features import (
    RhythmFeatures,
)
from reasoning_pipeline.domain.models.signal_quality import (
    SignalQuality,
)
from reasoning_pipeline.orchestration.ecg_analysis_pipeline import (
    ECGAnalysisPipeline,
)
from reasoning_pipeline.orchestration.model_input_preparer import (
    PreparedBeat,
)


def _signal() -> ECGSignal:
    return ECGSignal(
        record_id="record-001",
        samples=tuple(float(index) for index in range(500)),
        sampling_rate_hz=360.0,
        source="unit-test",
        lead_name="MLII",
    )


def _features() -> FeatureSet:
    return FeatureSet(
        signal_quality=SignalQuality(
            score=0.95,
            status=SignalQualityStatus.GOOD,
        ),
        r_peaks=RPeakSeries(
            sample_indices=(100, 300),
            timestamps_seconds=(100 / 360, 300 / 360),
            rr_intervals_ms=(200 / 360 * 1000,),
            detector_name="test",
            detector_version="1.0",
            confidence=0.95,
        ),
        rhythm=RhythmFeatures(
            heart_rate_mean_bpm=108.0,
            heart_rate_min_bpm=108.0,
            heart_rate_max_bpm=108.0,
            mean_rr_ms=555.56,
            sdnn_ms=0.0,
            rmssd_ms=None,
            pnn50_percent=None,
            irregularity_score=0.0,
        ),
        morphology=MorphologyFeatures(
            mean_qrs_duration_ms=90.0,
            mean_pr_interval_ms=None,
            mean_qt_interval_ms=None,
            mean_r_amplitude=1.0,
            abnormal_beat_count=0,
            morphology_confidence=0.9,
        ),
        extraction_version="test-extraction",
    )


def _prediction() -> ModelPrediction:
    return ModelPrediction(
        predicted_class=0,
        predicted_label="Normal Sinus Rhythm",
        probabilities=(0.8, 0.05, 0.05, 0.05, 0.05),
        confidence=0.8,
        checkpoint_path="/tmp/model.pth",
        checkpoint_hash="abc123",
        model_version="test-model",
        preprocessing_version="test-preprocessing",
    )


@dataclass
class FeatureExtractorStub:
    value: FeatureSet

    def extract(self, signal: ECGSignal) -> FeatureSet:
        assert signal.record_id == "record-001"
        return self.value


@dataclass
class InputPreparerStub:
    value: tuple[PreparedBeat, ...]

    def prepare_all(
        self,
        *,
        signal: ECGSignal,
        features: FeatureSet,
    ) -> tuple[PreparedBeat, ...]:
        assert signal.record_id == "record-001"
        assert features.extraction_version == "test-extraction"
        return self.value


@dataclass
class ClassifierStub:
    value: tuple[ModelPrediction, ...]

    def predict_many(self, beats):
        assert all(len(beat) == 216 for beat in beats)
        return self.value


class ExplainabilityServiceStub:
    def __init__(self) -> None:
        self.called = False

    def explain_recording(
        self,
        *,
        record_id,
        prepared_beats,
        beat_results,
    ):
        self.called = True
        assert record_id == "record-001"
        assert len(prepared_beats) == 2
        assert len(beat_results) == 2
        return None


class EvidenceBuilderStub:
    def build(
        self,
        record_id: str,
        prediction: ModelPrediction,
        features: FeatureSet,
    ) -> EvidenceBundle:
        return EvidenceBundle(
            record_id=record_id,
            prediction=prediction,
            features=features,
            supporting_evidence=(),
            conflicting_evidence=(),
            neutral_evidence=(),
            limitations=(),
            evidence_version="test-evidence",
        )


class ReasoningEngineStub:
    def reason(
        self,
        evidence: EvidenceBundle,
    ) -> ReasoningResult:
        return ReasoningResult(
            evidence=evidence,
            consistency_status=ConsistencyStatus.PARTIALLY_SUPPORTED,
            reasoning_confidence=0.7,
            conclusion="Test conclusion.",
            limitations=(),
            rule_trace=("test-rule",),
            reasoning_version="test-reasoning",
        )


class ReportGeneratorStub:
    def generate(
        self,
        reasoning_result: ReasoningResult,
    ) -> ClinicalReport:
        prediction = reasoning_result.evidence.prediction

        return ClinicalReport(
            record_id=reasoning_result.evidence.record_id,
            predicted_label=prediction.predicted_label,
            prediction_confidence=prediction.confidence,
            consistency_status=reasoning_result.consistency_status,
            reasoning_confidence=reasoning_result.reasoning_confidence,
            summary="Test clinical report.",
            supporting_findings=(),
            conflicting_findings=(),
            limitations=(),
            recommended_action="Manual review.",
            model_version=prediction.model_version,
            preprocessing_version=prediction.preprocessing_version,
            evidence_version=(
                reasoning_result.evidence.evidence_version
            ),
            reasoning_version=reasoning_result.reasoning_version,
            report_version="test-report",
            disclaimer="Research use only.",
        )


class NarrativeGeneratorStub:
    def generate(
        self,
        report: ClinicalReport,
    ) -> NarrativeResult:
        return NarrativeResult(
            record_id=report.record_id,
            doctor_report="Test doctor report.",
            next_of_kin_summary="Test family summary.",
            provider="test-provider",
            model_name="test-model",
            prompt_version="test-prompt",
            fallback_used=False,
        )


def test_pipeline_returns_complete_analysis_result() -> None:
    signal = _signal()
    features = _features()
    prediction = _prediction()

    prepared_beats = (
        PreparedBeat(
            beat_index=0,
            r_peak_sample_index=100,
            source_start_sample_index=28,
            source_stop_sample_index_exclusive=244,
            r_peak_timestamp_seconds=100 / 360,
            source_start_timestamp_seconds=28 / 360,
            source_stop_timestamp_seconds_exclusive=244 / 360,
            sampling_rate_hz=360.0,
            samples=tuple(0.0 for _ in range(216)),
        ),
        PreparedBeat(
            beat_index=1,
            r_peak_sample_index=300,
            source_start_sample_index=228,
            source_stop_sample_index_exclusive=444,
            r_peak_timestamp_seconds=300 / 360,
            source_start_timestamp_seconds=228 / 360,
            source_stop_timestamp_seconds_exclusive=444 / 360,
            sampling_rate_hz=360.0,
            samples=tuple(0.0 for _ in range(216)),
        ),
    )
    explainability_service = ExplainabilityServiceStub()

    pipeline = ECGAnalysisPipeline(
        feature_extractor=FeatureExtractorStub(features),
        model_input_preparer=InputPreparerStub(prepared_beats),
        classifier=ClassifierStub((prediction, prediction)),
        evidence_builder=EvidenceBuilderStub(),
        reasoning_engine=ReasoningEngineStub(),
        report_generator=ReportGeneratorStub(),
        narrative_generator=NarrativeGeneratorStub(),
        explainability_service=explainability_service,
    )

    result = pipeline.analyse(signal)

    assert result.record_id == "record-001"
    assert result.signal is signal
    assert result.features is features
    assert result.prepared_beat is prepared_beats[1]
    assert result.prediction is prediction
    assert result.recording_summary.total_valid_beats == 2
    assert result.recording_summary.class_counts == (
        ("N", 2),
        ("S", 0),
        ("V", 0),
        ("F", 0),
        ("Q", 0),
    )
    assert result.recording_summary.abnormal_beat_count == 0
    assert result.recording_summary.abnormal_beat_percentage == 0.0
    assert result.recording_summary.dominant_predicted_label == "N"
    assert result.recording_explanation is None
    assert result.recording_attribution_overlay is None
    assert result.analysis_scope is AnalysisScope.EXPLORATORY_EXTERNAL_SOURCE
    assert explainability_service.called
    assert tuple(
        result.beat_index
        for result in result.recording_summary.beat_results
    ) == (0, 1)
    assert result.evidence.prediction is prediction
    assert result.reasoning.evidence is result.evidence
    assert result.clinical_report.predicted_label == (
        "Normal Sinus Rhythm"
    )
    assert result.narrative.doctor_report == "Test doctor report."

    api_response = AnalysisResponse.from_domain(result)
    api_payload = api_response.model_dump(mode="json")
    assert api_payload["recording_explanation"] is None
    assert api_payload["recording_attribution_overlay"] is None

    legacy_client_fields = {
        "signal",
        "prediction",
        "recording_summary",
        "evidence",
        "reasoning",
        "clinical_report",
        "narrative",
    }
    assert legacy_client_fields <= api_payload.keys()
