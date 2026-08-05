from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from reasoning_pipeline.domain.models.beat_analysis_result import (
    BeatAnalysisResult,
)
from reasoning_pipeline.domain.models.recording_explanation import (
    RecordingExplanation,
)

if TYPE_CHECKING:
    from reasoning_pipeline.orchestration.model_input_preparer import PreparedBeat


class ExplainabilityServiceProtocol(Protocol):
    def explain_recording(
        self,
        *,
        record_id: str,
        prepared_beats: tuple[PreparedBeat, ...],
        beat_results: tuple[BeatAnalysisResult, ...],
    ) -> RecordingExplanation | None:
        ...
