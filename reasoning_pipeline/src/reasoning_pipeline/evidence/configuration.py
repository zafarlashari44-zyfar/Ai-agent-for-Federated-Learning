from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class EvidenceBuilderConfiguration:
    normal_heart_rate_min_bpm: float = 60.0
    normal_heart_rate_max_bpm: float = 100.0
    irregularity_threshold: float = 0.50
    wide_qrs_threshold_ms: float = 120.0
    low_model_confidence_threshold: float = 0.60
    low_signal_quality_threshold: float = 0.50
    evidence_version: str = "evidence-builder-v1"

    def __post_init__(self) -> None:
        numeric_values = (
            self.normal_heart_rate_min_bpm,
            self.normal_heart_rate_max_bpm,
            self.irregularity_threshold,
            self.wide_qrs_threshold_ms,
            self.low_model_confidence_threshold,
            self.low_signal_quality_threshold,
        )

        if not all(isfinite(value) for value in numeric_values):
            raise ValueError("configuration values must be finite")

        if self.normal_heart_rate_min_bpm <= 0:
            raise ValueError(
                "normal_heart_rate_min_bpm must be greater than zero"
            )

        if (
            self.normal_heart_rate_max_bpm
            <= self.normal_heart_rate_min_bpm
        ):
            raise ValueError(
                "normal_heart_rate_max_bpm must be greater than "
                "normal_heart_rate_min_bpm"
            )

        if not 0.0 <= self.irregularity_threshold <= 1.0:
            raise ValueError(
                "irregularity_threshold must be between zero and one"
            )

        if self.wide_qrs_threshold_ms <= 0:
            raise ValueError(
                "wide_qrs_threshold_ms must be greater than zero"
            )

        if not 0.0 <= self.low_model_confidence_threshold <= 1.0:
            raise ValueError(
                "low_model_confidence_threshold must be between "
                "zero and one"
            )

        if not 0.0 <= self.low_signal_quality_threshold <= 1.0:
            raise ValueError(
                "low_signal_quality_threshold must be between "
                "zero and one"
            )

        if not self.evidence_version.strip():
            raise ValueError("evidence_version cannot be empty")
