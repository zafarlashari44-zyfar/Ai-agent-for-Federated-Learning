from dataclasses import dataclass

from reasoning_pipeline.domain.enums.statuses import SignalSuitabilityStatus


@dataclass(frozen=True)
class SignalSuitabilityAssessment:
    status: SignalSuitabilityStatus
    suitable_for_processing: bool
    quality_score: float
    duration_seconds: float
    sampling_rate_hz: float
    selected_lead: str | None
    units: str | None
    detected_r_peak_count: int
    estimated_heart_rate_bpm: float | None
    finite_sample_ratio: float
    flatline_percentage: float
    clipping_percentage: float
    noise_score: float
    warnings: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("quality_score must be between zero and one")
        if self.status is SignalSuitabilityStatus.REJECTED:
            if self.suitable_for_processing or not self.rejection_reasons:
                raise ValueError("Rejected suitability requires reasons")
