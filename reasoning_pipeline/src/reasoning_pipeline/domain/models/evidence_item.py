from dataclasses import dataclass

from reasoning_pipeline.domain.enums.statuses import EvidenceDirection


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    feature_name: str
    measured_value: float | int | str | None
    unit: str | None
    interpretation: str
    direction: EvidenceDirection
    reliability: float
    source_reference: str

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise ValueError("evidence_id cannot be empty")

        if not self.feature_name.strip():
            raise ValueError("feature_name cannot be empty")

        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("reliability must be between zero and one")
