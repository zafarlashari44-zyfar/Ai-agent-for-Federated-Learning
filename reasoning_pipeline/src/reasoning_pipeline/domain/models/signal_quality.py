from dataclasses import dataclass

from reasoning_pipeline.domain.enums.statuses import SignalQualityStatus


@dataclass(frozen=True)
class SignalQuality:
    score: float
    status: SignalQualityStatus
    noise_score: float | None = None
    valid_sample_ratio: float | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be between zero and one")

        if self.noise_score is not None:
            if not 0.0 <= self.noise_score <= 1.0:
                raise ValueError(
                    "noise_score must be between zero and one"
                )

        if self.valid_sample_ratio is not None:
            if not 0.0 <= self.valid_sample_ratio <= 1.0:
                raise ValueError(
                    "valid_sample_ratio must be between zero and one"
                )
