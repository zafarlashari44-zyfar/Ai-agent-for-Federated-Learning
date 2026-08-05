from reasoning_pipeline.domain.models.beat_analysis_result import (
    BeatAnalysisResult,
)


class ExplainAbnormalBeatsPolicy:
    """Select every non-normal AAMI beat for explanation."""

    NORMAL_CLASS_INDEX = 0

    @property
    def policy_id(self) -> str:
        return "explain-abnormal-beats"

    def select(
        self,
        beat_results: tuple[BeatAnalysisResult, ...],
    ) -> tuple[int, ...]:
        return tuple(
            result.beat_index
            for result in beat_results
            if (
                result.prediction.predicted_class
                != self.NORMAL_CLASS_INDEX
            )
        )
