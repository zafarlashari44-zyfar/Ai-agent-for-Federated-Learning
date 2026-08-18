from dataclasses import dataclass


@dataclass(frozen=True)
class ReportConfiguration:
    report_version: str = "clinical-report-v1"
    disclaimer: str = (
        "This automated report is intended for research and decision-support "
        "purposes only. It must not replace assessment by a qualified "
        "healthcare professional."
    )

    def __post_init__(self) -> None:
        if not self.report_version.strip():
            raise ValueError("report_version cannot be empty")

        if not self.disclaimer.strip():
            raise ValueError("disclaimer cannot be empty")
