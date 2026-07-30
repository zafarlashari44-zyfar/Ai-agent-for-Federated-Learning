from dataclasses import dataclass

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
from reasoning_pipeline.domain.models.recording_analysis_summary import (
    RecordingAnalysisSummary,
)
from reasoning_pipeline.domain.models.recording_attribution_overlay import (
    RecordingAttributionOverlay,
)
from reasoning_pipeline.domain.models.recording_explanation import (
    RecordingExplanation,
)
from reasoning_pipeline.orchestration.model_input_preparer import (
    PreparedBeat,
)


@dataclass(frozen=True)
class ECGAnalysisResult:
    """
    Complete output produced by one end-to-end ECG analysis.

    This result deliberately retains each major intermediate artefact so
    the API, dashboard, tests, and research evaluation can inspect the
    model prediction and deterministic reasoning separately.
    """

    signal: ECGSignal
    features: FeatureSet
    prepared_beat: PreparedBeat
    prediction: ModelPrediction
    recording_summary: RecordingAnalysisSummary
    evidence: EvidenceBundle
    reasoning: ReasoningResult
    clinical_report: ClinicalReport
    narrative: NarrativeResult
    recording_explanation: RecordingExplanation | None = None
    recording_attribution_overlay: RecordingAttributionOverlay | None = None

    def __post_init__(self) -> None:
        record_ids = {
            self.signal.record_id,
            self.evidence.record_id,
            self.clinical_report.record_id,
            self.narrative.record_id,
        }

        if len(record_ids) != 1:
            raise ValueError(
                "All ECG analysis artefacts must share the same record_id."
            )

    @property
    def record_id(self) -> str:
        return self.signal.record_id
