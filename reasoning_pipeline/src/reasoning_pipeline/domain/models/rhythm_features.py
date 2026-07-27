from dataclasses import dataclass


@dataclass(frozen=True)
class RhythmFeatures:
    heart_rate_mean_bpm: float | None
    heart_rate_min_bpm: float | None
    heart_rate_max_bpm: float | None
    mean_rr_ms: float | None
    sdnn_ms: float | None
    rmssd_ms: float | None
    pnn50_percent: float | None
    irregularity_score: float | None

    def __post_init__(self) -> None:
        if self.irregularity_score is not None:
            if not 0.0 <= self.irregularity_score <= 1.0:
                raise ValueError(
                    "irregularity_score must be between zero and one"
                )
