from dataclasses import dataclass

from reasoning_pipeline.domain.enums.statuses import ConsistencyStatus


@dataclass(frozen=True)
class ClinicalReport:
    record_id: str
    predicted_label: str
    prediction_confidence: float
    consistency_status: ConsistencyStatus
    reasoning_confidence: float
    summary: str
    supporting_findings: tuple[str, ...]
    conflicting_findings: tuple[str, ...]
    limitations: tuple[str, ...]
    recommended_action: str
    model_version: str
    preprocessing_version: str
    evidence_version: str
    reasoning_version: str
    report_version: str
    disclaimer: str

    def __post_init__(self) -> None:
        required_text_values = (
            self.record_id,
            self.predicted_label,
            self.summary,
            self.recommended_action,
            self.model_version,
            self.preprocessing_version,
            self.evidence_version,
            self.reasoning_version,
            self.report_version,
            self.disclaimer,
        )

        if any(not value.strip() for value in required_text_values):
            raise ValueError(
                "required clinical report text fields cannot be empty"
            )

        if not 0.0 <= self.prediction_confidence <= 1.0:
            raise ValueError(
                "prediction_confidence must be between zero and one"
            )

        if not 0.0 <= self.reasoning_confidence <= 1.0:
            raise ValueError(
                "reasoning_confidence must be between zero and one"
            )
