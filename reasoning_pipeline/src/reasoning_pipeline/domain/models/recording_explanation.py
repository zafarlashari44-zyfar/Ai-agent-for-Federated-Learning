from dataclasses import dataclass

from reasoning_pipeline.domain.models.beat_explanation import BeatExplanation


@dataclass(frozen=True)
class RecordingExplanation:
    """Ordered explainability output for one ECG recording."""

    record_id: str
    selection_policy: str
    total_valid_beats: int
    total_explained_beats: int
    beat_explanations: tuple[BeatExplanation, ...]
    requested_methods: tuple[str, ...]
    completed_methods: tuple[str, ...]
    model_version: str
    checkpoint_hash: str
    preprocessing_version: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("record_id", self.record_id),
            ("selection_policy", self.selection_policy),
            ("model_version", self.model_version),
            ("checkpoint_hash", self.checkpoint_hash),
            ("preprocessing_version", self.preprocessing_version),
        ):
            if not value.strip():
                raise ValueError(f"{name} cannot be empty")

        if self.total_valid_beats < 0:
            raise ValueError("total_valid_beats cannot be negative")
        if self.total_explained_beats != len(self.beat_explanations):
            raise ValueError(
                "total_explained_beats must match beat_explanations"
            )
        if self.total_explained_beats > self.total_valid_beats:
            raise ValueError(
                "total_explained_beats cannot exceed total_valid_beats"
            )

        beat_indices = tuple(
            explanation.beat_index
            for explanation in self.beat_explanations
        )
        if beat_indices != tuple(sorted(beat_indices)):
            raise ValueError("Beat explanations must be ordered by beat_index")
        if len(set(beat_indices)) != len(beat_indices):
            raise ValueError("Beat explanation indices must be unique")
        if len(set(self.requested_methods)) != len(self.requested_methods):
            raise ValueError("requested_methods must be unique")
        if len(set(self.completed_methods)) != len(self.completed_methods):
            raise ValueError("completed_methods must be unique")
