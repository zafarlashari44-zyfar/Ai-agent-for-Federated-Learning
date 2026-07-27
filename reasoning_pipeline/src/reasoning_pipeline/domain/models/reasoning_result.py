from dataclasses import dataclass
from typing import Tuple

from reasoning_pipeline.domain.enums.statuses import ConsistencyStatus
from reasoning_pipeline.domain.models.evidence_bundle import EvidenceBundle


@dataclass(frozen=True)
class ReasoningResult:
    evidence: EvidenceBundle
    consistency_status: ConsistencyStatus
    reasoning_confidence: float
    conclusion: str
    limitations: Tuple[str, ...]
    rule_trace: Tuple[str, ...]
    reasoning_version: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.reasoning_confidence <= 1.0:
            raise ValueError(
                "reasoning_confidence must be between zero and one"
            )

        if not self.conclusion.strip():
            raise ValueError("conclusion cannot be empty")

        if not self.reasoning_version.strip():
            raise ValueError("reasoning_version cannot be empty")
