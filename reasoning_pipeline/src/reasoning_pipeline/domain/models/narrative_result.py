from dataclasses import dataclass


@dataclass(frozen=True)
class NarrativeResult:
    record_id: str
    doctor_report: str
    next_of_kin_summary: str
    provider: str
    model_name: str
    prompt_version: str
    fallback_used: bool
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        required_values = (
            self.record_id,
            self.doctor_report,
            self.next_of_kin_summary,
            self.provider,
            self.model_name,
            self.prompt_version,
        )

        if any(not value.strip() for value in required_values):
            raise ValueError(
                "required narrative result text fields cannot be empty"
            )
