from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from reasoning_pipeline.application.ports.explainability_service import (
    ExplainabilityServiceProtocol,
)
from reasoning_pipeline.application.ports.ood_assessor import OODAssessorProtocol
from reasoning_pipeline.application.ports.recording_attribution_compositor import (
    RecordingAttributionCompositorProtocol,
)
from reasoning_pipeline.application.services.explainability_service import (
    ExplainabilityService,
)
from reasoning_pipeline.baseline_adapter.classifier import (
    BaselineClassifier,
)
from reasoning_pipeline.domain.enums.statuses import AnalysisScope, OODStatus
from reasoning_pipeline.domain.models.beat_analysis_result import (
    BeatAnalysisResult,
)
from reasoning_pipeline.domain.models.clinical_report import (
    ClinicalReport,
)
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.domain.models.evidence_bundle import (
    EvidenceBundle,
)
from reasoning_pipeline.domain.models.feature_set import FeatureSet
from reasoning_pipeline.domain.models.model_prediction import ModelPrediction
from reasoning_pipeline.domain.models.narrative_result import (
    NarrativeResult,
)
from reasoning_pipeline.domain.models.reasoning_result import (
    ReasoningResult,
)
from reasoning_pipeline.domain.models.recording_analysis_summary import (
    RecordingAnalysisSummary,
)
from reasoning_pipeline.domain.models.signal_suitability_assessment import (
    SignalSuitabilityAssessment,
)
from reasoning_pipeline.evidence.builder import EvidenceBuilder
from reasoning_pipeline.infrastructure.explainability import (
    RecordingAttributionCompositor,
)
from reasoning_pipeline.infrastructure.explainability.grad_cam_1d import (
    GradCAM1D,
)
from reasoning_pipeline.infrastructure.explainability.policies import (
    ExplainAbnormalBeatsPolicy,
)
from reasoning_pipeline.infrastructure.explainability.source_attribution_mapper import (
    SourceAttributionMapper,
)
from reasoning_pipeline.infrastructure.ood_assessment import HeuristicOODAssessor
from reasoning_pipeline.narrative.generator import NarrativeGenerator
from reasoning_pipeline.orchestration.analysis_result import (
    ECGAnalysisResult,
)
from reasoning_pipeline.orchestration.model_input_preparer import (
    ModelInputPreparer,
    PreparedBeat,
)
from reasoning_pipeline.reasoning.engine import ReasoningEngine
from reasoning_pipeline.reporting.generator import ReportGenerator
from reasoning_pipeline.scribe_v2.feature_extraction_service import (
    ScribeV2FeatureExtractionService,
)
from reasoning_pipeline.scribe_v2.morphology_feature_extractor import (
    MorphologyFeatureExtractor,
)
from reasoning_pipeline.scribe_v2.service import ScribeV2InputService


class FeatureExtractionServiceProtocol(Protocol):
    def extract(self, signal: ECGSignal) -> FeatureSet:
        ...


class ModelInputPreparerProtocol(Protocol):
    def prepare_all(
        self,
        *,
        signal: ECGSignal,
        features: FeatureSet,
    ) -> tuple[PreparedBeat, ...]:
        ...


class ClassifierProtocol(Protocol):
    def predict_many(
        self,
        beats: Sequence[Sequence[float] | NDArray[np.float32]],
    ) -> tuple[ModelPrediction, ...]:
        ...


class EvidenceBuilderProtocol(Protocol):
    def build(
        self,
        record_id: str,
        prediction: ModelPrediction,
        features: FeatureSet,
    ) -> EvidenceBundle:
        ...


class ReasoningEngineProtocol(Protocol):
    def reason(
        self,
        evidence: EvidenceBundle,
    ) -> ReasoningResult:
        ...


class ReportGeneratorProtocol(Protocol):
    def generate(
        self,
        reasoning_result: ReasoningResult,
    ) -> ClinicalReport:
        ...


class NarrativeGeneratorProtocol(Protocol):
    def generate(
        self,
        report: ClinicalReport,
    ) -> NarrativeResult:
        ...


class ECGAnalysisPipeline:
    """
    Coordinate the complete ECG reasoning workflow.

    The orchestration layer does not implement signal processing, model
    architecture, evidence rules, or narrative generation. It only joins
    the independently testable components in the required order.
    """

    def __init__(
        self,
        *,
        feature_extractor: FeatureExtractionServiceProtocol,
        model_input_preparer: ModelInputPreparerProtocol,
        classifier: ClassifierProtocol,
        evidence_builder: EvidenceBuilderProtocol,
        reasoning_engine: ReasoningEngineProtocol,
        report_generator: ReportGeneratorProtocol,
        narrative_generator: NarrativeGeneratorProtocol,
        explainability_service: ExplainabilityServiceProtocol | None = None,
        recording_attribution_compositor: (
            RecordingAttributionCompositorProtocol | None
        ) = None,
        recording_attribution_method: str = "grad-cam-1d",
        ood_assessor: OODAssessorProtocol | None = None,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.model_input_preparer = model_input_preparer
        self.classifier = classifier
        self.evidence_builder = evidence_builder
        self.reasoning_engine = reasoning_engine
        self.report_generator = report_generator
        self.narrative_generator = narrative_generator
        self.explainability_service = explainability_service
        self.recording_attribution_compositor = (
            recording_attribution_compositor
        )
        self.recording_attribution_method = recording_attribution_method
        self.ood_assessor = ood_assessor

    def analyse(
        self,
        signal: ECGSignal,
        *,
        suitability_assessment: SignalSuitabilityAssessment | None = None,
    ) -> ECGAnalysisResult:
        """
        Analyse one validated ECG signal from feature extraction through
        narrative generation.
        """
        features = self.feature_extractor.extract(signal)

        prepared_beats = self.model_input_preparer.prepare_all(
            signal=signal,
            features=features,
        )

        predictions = self.classifier.predict_many(
            tuple(beat.samples for beat in prepared_beats)
        )

        if len(predictions) != len(prepared_beats):
            raise RuntimeError(
                "Classifier must return one prediction per prepared beat"
            )

        beat_results = tuple(
            BeatAnalysisResult(
                beat_index=beat.beat_index,
                r_peak_sample_index=beat.r_peak_sample_index,
                source_start_sample_index=beat.source_start_sample_index,
                source_stop_sample_index_exclusive=(
                    beat.source_stop_sample_index_exclusive
                ),
                r_peak_timestamp_seconds=beat.r_peak_timestamp_seconds,
                source_start_timestamp_seconds=(
                    beat.source_start_timestamp_seconds
                ),
                source_stop_timestamp_seconds_exclusive=(
                    beat.source_stop_timestamp_seconds_exclusive
                ),
                sampling_rate_hz=beat.sampling_rate_hz,
                prediction=prediction,
            )
            for beat, prediction in zip(
                prepared_beats,
                predictions,
                strict=True,
            )
        )
        recording_summary = RecordingAnalysisSummary.from_beat_results(
            beat_results
        )

        ood_assessment = None
        if self.ood_assessor is not None and suitability_assessment is not None:
            ood_assessment = self.ood_assessor.assess(
                signal=signal,
                predictions=predictions,
                suitability=suitability_assessment,
            )

        recording_explanation = None
        if self.explainability_service is not None:
            recording_explanation = (
                self.explainability_service.explain_recording(
                    record_id=signal.record_id,
                    prepared_beats=prepared_beats,
                    beat_results=beat_results,
                )
            )

        recording_attribution_overlay = None
        if (
            recording_explanation is not None
            and self.recording_attribution_compositor is not None
        ):
            recording_attribution_overlay = (
                self.recording_attribution_compositor.compose(
                    total_source_samples=signal.sample_count,
                    sampling_rate_hz=signal.sampling_rate_hz,
                    recording_explanation=recording_explanation,
                    method_id=self.recording_attribution_method,
                )
            )

        representative_index = len(prepared_beats) // 2
        prepared_beat = prepared_beats[representative_index]
        prediction = predictions[representative_index]

        evidence = self.evidence_builder.build(
            record_id=signal.record_id,
            prediction=prediction,
            features=features,
        )

        reasoning = self.reasoning_engine.reason(evidence)

        clinical_report = self.report_generator.generate(
            reasoning
        )

        narrative = self.narrative_generator.generate(
            clinical_report
        )

        analysis_scope = self._analysis_scope(signal, suitability_assessment)
        external = analysis_scope is AnalysisScope.EXPLORATORY_EXTERNAL_SOURCE
        model_scope_statement = (
            "Predictions are limited to the MIT-BIH AAMI classes "
            "N, S, V, F and Q."
        )
        recommended_interpretation = (
            "This external recording is outside the model's validated "
            "training source, so results are exploratory."
            if external
            else "Interpret results within the validated MIT-BIH-compatible scope."
        )
        analysis_warnings = list(
            suitability_assessment.warnings
            if suitability_assessment is not None
            else ()
        )
        if external:
            analysis_warnings.append(
                "External-source predictions are exploratory, not clinically validated."
            )
        if (
            ood_assessment is not None
            and ood_assessment.status is OODStatus.LIKELY_OUT_OF_DISTRIBUTION
        ):
            analysis_warnings.extend(ood_assessment.warnings)

        return ECGAnalysisResult(
            signal=signal,
            features=features,
            prepared_beat=prepared_beat,
            prediction=prediction,
            recording_summary=recording_summary,
            evidence=evidence,
            reasoning=reasoning,
            clinical_report=clinical_report,
            narrative=narrative,
            recording_explanation=recording_explanation,
            recording_attribution_overlay=recording_attribution_overlay,
            signal_suitability=suitability_assessment,
            ood_assessment=ood_assessment,
            analysis_scope=analysis_scope,
            model_scope_statement=model_scope_statement,
            recommended_interpretation=recommended_interpretation,
            analysis_warnings=tuple(dict.fromkeys(analysis_warnings)),
        )

    @staticmethod
    def _analysis_scope(
        signal: ECGSignal,
        suitability: SignalSuitabilityAssessment | None,
    ) -> AnalysisScope:
        if suitability is not None and not suitability.suitable_for_processing:
            return AnalysisScope.UNSUPPORTED
        mit_bih_compatible = "mit-bih" in signal.source.casefold() or (
            signal.source_format == "npy"
            and signal.lead_name in {None, "MLII", "II", "Lead II"}
        )
        if mit_bih_compatible:
            return AnalysisScope.VALIDATED_MIT_BIH_COMPATIBLE
        return AnalysisScope.EXPLORATORY_EXTERNAL_SOURCE

    def analyse_npy(
        self,
        *,
        input_service: ScribeV2InputService,
        file_path: str | Path,
        sampling_rate_hz: float,
        record_id: str | None = None,
        source: str = "npy",
        lead_name: str | None = None,
    ) -> ECGAnalysisResult:
        """
        Load, validate, and analyse one NumPy ECG file.
        """
        signal = input_service.load_npy(
            file_path=file_path,
            sampling_rate_hz=sampling_rate_hz,
            record_id=record_id,
            source=source,
            lead_name=lead_name,
        )

        return self.analyse(signal)


def create_default_pipeline(
    *,
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> ECGAnalysisPipeline:
    """
    Construct the production pipeline using the project's default
    implementations.
    """
    feature_extractor = ScribeV2FeatureExtractionService(
        morphology_extractor=MorphologyFeatureExtractor(),
    )
    classifier = BaselineClassifier(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    explainers = (
        GradCAM1D(
            model=classifier.model,
            target_layer=classifier.model.features[8],
            target_layer_name="features.8",
        ),
    )

    return ECGAnalysisPipeline(
        feature_extractor=feature_extractor,
        model_input_preparer=ModelInputPreparer(),
        classifier=classifier,
        evidence_builder=EvidenceBuilder(),
        reasoning_engine=ReasoningEngine(),
        report_generator=ReportGenerator(),
        narrative_generator=NarrativeGenerator(),
        explainability_service=ExplainabilityService(
            explainers=explainers,
            mapper=SourceAttributionMapper(),
            selection_policy=ExplainAbnormalBeatsPolicy(),
        ),
        recording_attribution_compositor=RecordingAttributionCompositor(),
        ood_assessor=HeuristicOODAssessor(),
    )
