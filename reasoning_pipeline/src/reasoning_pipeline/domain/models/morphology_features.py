from dataclasses import dataclass


@dataclass(frozen=True)
class MorphologyFeatures:
    mean_qrs_duration_ms: float | None
    mean_pr_interval_ms: float | None
    mean_qt_interval_ms: float | None
    mean_r_amplitude: float | None
    abnormal_beat_count: int | None
    morphology_confidence: float

    def __post_init__(self) -> None:
        if self.abnormal_beat_count is not None:
            if self.abnormal_beat_count < 0:
                raise ValueError(
                    "abnormal_beat_count cannot be negative"
                )

        if not 0.0 <= self.morphology_confidence <= 1.0:
            raise ValueError(
                "morphology_confidence must be between zero and one"
            )
