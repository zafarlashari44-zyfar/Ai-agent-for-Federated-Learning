from dataclasses import dataclass
from typing import Tuple

from reasoning_pipeline.domain.models.evidence_item import EvidenceItem
from reasoning_pipeline.domain.models.feature_set import FeatureSet
from reasoning_pipeline.domain.models.model_prediction import (
    ModelPrediction,
)


@dataclass(frozen=True)
class EvidenceBundle:
    record_id: str
    prediction: ModelPrediction
    features: FeatureSet
    supporting_evidence: Tuple[EvidenceItem, ...]
    conflicting_evidence: Tuple[EvidenceItem, ...]
    neutral_evidence: Tuple[EvidenceItem, ...]
    limitations: Tuple[str, ...]
    evidence_version: str

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise ValueError("record_id cannot be empty")

        if not self.evidence_version.strip():
            raise ValueError("evidence_version cannot be empty")
