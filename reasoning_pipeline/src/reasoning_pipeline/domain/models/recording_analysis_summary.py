from dataclasses import dataclass

from reasoning_pipeline.baseline_adapter.labels import CLASS_LABELS
from reasoning_pipeline.domain.models.beat_analysis_result import (
    BeatAnalysisResult,
)


@dataclass(frozen=True)
class RecordingAnalysisSummary:
    """Ordered beat predictions and aggregate AAMI class statistics."""

    total_valid_beats: int
    class_counts: tuple[tuple[str, int], ...]
    abnormal_beat_count: int
    abnormal_beat_percentage: float
    dominant_predicted_class: int
    dominant_predicted_label: str
    beat_results: tuple[BeatAnalysisResult, ...]

    @classmethod
    def from_beat_results(
        cls,
        beat_results: tuple[BeatAnalysisResult, ...],
    ) -> "RecordingAnalysisSummary":
        if not beat_results:
            raise ValueError("Recording analysis requires at least one beat")

        counts = {
            class_index: 0
            for class_index in CLASS_LABELS
        }
        for result in beat_results:
            counts[result.prediction.predicted_class] += 1

        dominant_class = max(
            counts,
            key=lambda class_index: counts[class_index],
        )
        total = len(beat_results)
        abnormal_count = total - counts[0]

        return cls(
            total_valid_beats=total,
            class_counts=tuple(
                (CLASS_LABELS[class_index], counts[class_index])
                for class_index in CLASS_LABELS
            ),
            abnormal_beat_count=abnormal_count,
            abnormal_beat_percentage=abnormal_count / total * 100.0,
            dominant_predicted_class=dominant_class,
            dominant_predicted_label=CLASS_LABELS[dominant_class],
            beat_results=beat_results,
        )

    def __post_init__(self) -> None:
        if self.total_valid_beats != len(self.beat_results):
            raise ValueError(
                "total_valid_beats must match the number of beat results"
            )

        if sum(count for _, count in self.class_counts) != self.total_valid_beats:
            raise ValueError("AAMI class counts must sum to total_valid_beats")

        expected_labels = tuple(CLASS_LABELS.values())
        if tuple(label for label, _ in self.class_counts) != expected_labels:
            raise ValueError("class_counts must use AAMI class order N, S, V, F, Q")
