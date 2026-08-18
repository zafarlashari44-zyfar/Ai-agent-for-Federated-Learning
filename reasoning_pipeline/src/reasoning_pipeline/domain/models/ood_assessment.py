from dataclasses import dataclass

from reasoning_pipeline.domain.enums.statuses import OODStatus


@dataclass(frozen=True)
class OODAssessment:
    status: OODStatus
    heuristic_score: int
    maximum_class_probability: float
    normalized_prediction_entropy: float
    q_class_proportion: float
    low_confidence_beat_proportion: float
    probability_instability: float
    indicators: tuple[str, ...]
    reasons: tuple[str, ...]
    warnings: tuple[str, ...] = ()
