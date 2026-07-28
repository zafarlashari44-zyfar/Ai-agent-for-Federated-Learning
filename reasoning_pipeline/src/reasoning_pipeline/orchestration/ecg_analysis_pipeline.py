from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from reasoning_pipeline.baseline_adapter.classifier import (
    BaselineClassifier,
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
from reasoning_pipeline.domain.models.narrative_result import (
    NarrativeResult,
)
from reasoning_pipeline.domain.models.reasoning_result import (
    ReasoningResult,
)
from reasoning_pipeline.evidence.builder import EvidenceBuilder
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
    def prepare_representative(
        self,
        *,
        signal: ECGSignal,
        features: FeatureSet,
    ) -> PreparedBeat:
        ...


class ClassifierProtocol(Protocol):
    def predict(
        self,
        beat: Sequence[float] | NDArray[np.float32],
    ) -> ModelPrediction:
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
    ) -> None:
        self.feature_extractor = feature_extractor
        self.model_input_preparer = model_input_preparer
        self.classifier = classifier
        self.evidence_builder = evidence_builder
        self.reasoning_engine = reasoning_engine
        self.report_generator = report_generator
        self.narrative_generator = narrative_generator

    def analyse(
        self,
        signal: ECGSignal,
    ) -> ECGAnalysisResult:
        """
        Analyse one validated ECG signal from feature extraction through
        narrative generation.
        """
        features = self.feature_extractor.extract(signal)

        prepared_beat = (
            self.model_input_preparer.prepare_representative(
                signal=signal,
                features=features,
            )
        )

        prediction = self.classifier.predict(
            prepared_beat.samples
        )

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

        return ECGAnalysisResult(
            signal=signal,
            features=features,
            prepared_beat=prepared_beat,
            prediction=prediction,
            evidence=evidence,
            reasoning=reasoning,
            clinical_report=clinical_report,
            narrative=narrative,
        )

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

    return ECGAnalysisPipeline(
        feature_extractor=feature_extractor,
        model_input_preparer=ModelInputPreparer(),
        classifier=BaselineClassifier(
            checkpoint_path=checkpoint_path,
            device=device,
        ),
        evidence_builder=EvidenceBuilder(),
        reasoning_engine=ReasoningEngine(),
        report_generator=ReportGenerator(),
        narrative_generator=NarrativeGenerator(),
    )
