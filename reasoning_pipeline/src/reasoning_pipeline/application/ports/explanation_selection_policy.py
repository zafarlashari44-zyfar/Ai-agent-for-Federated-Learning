from typing import Protocol

from reasoning_pipeline.domain.models.beat_analysis_result import (
    BeatAnalysisResult,
)


class ExplanationSelectionPolicyProtocol(Protocol):
    @property
    def policy_id(self) -> str:
        ...

    def select(
        self,
        beat_results: tuple[BeatAnalysisResult, ...],
    ) -> tuple[int, ...]:
        """Return selected original beat indices."""
        ...
