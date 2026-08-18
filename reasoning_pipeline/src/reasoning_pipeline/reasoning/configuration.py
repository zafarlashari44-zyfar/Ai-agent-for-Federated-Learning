from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ReasoningConfiguration:
    strong_support_threshold: float = 0.60
    partial_support_threshold: float = 0.15
    conflict_ratio_threshold: float = 0.35
    low_signal_quality_threshold: float = 0.50
    low_reliability_threshold: float = 0.50
    conflict_penalty: float = 0.50
    insufficient_evidence_factor: float = 0.50
    low_signal_quality_factor: float = 0.40
    minimum_strong_support_items: int = 2
    reasoning_version: str = "reasoning-engine-v1"

    def __post_init__(self) -> None:
        numeric_values = (
            self.strong_support_threshold,
            self.partial_support_threshold,
            self.conflict_ratio_threshold,
            self.low_signal_quality_threshold,
            self.low_reliability_threshold,
            self.conflict_penalty,
            self.insufficient_evidence_factor,
            self.low_signal_quality_factor,
        )

        if not all(isfinite(value) for value in numeric_values):
            raise ValueError("configuration values must be finite")

        bounded_values = (
            self.strong_support_threshold,
            self.partial_support_threshold,
            self.conflict_ratio_threshold,
            self.low_signal_quality_threshold,
            self.low_reliability_threshold,
            self.conflict_penalty,
            self.insufficient_evidence_factor,
            self.low_signal_quality_factor,
        )

        if any(value < 0.0 or value > 1.0 for value in bounded_values):
            raise ValueError(
                "configuration thresholds must be between zero and one"
            )

        if (
            self.strong_support_threshold
            <= self.partial_support_threshold
        ):
            raise ValueError(
                "strong_support_threshold must be greater than "
                "partial_support_threshold"
            )

        if self.minimum_strong_support_items < 1:
            raise ValueError(
                "minimum_strong_support_items must be at least one"
            )

        if not self.reasoning_version.strip():
            raise ValueError("reasoning_version cannot be empty")
