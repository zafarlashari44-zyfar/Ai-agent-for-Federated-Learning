from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class NarrativeConfiguration:
    model_name: str = "llama3.2:3b"
    host: str = "http://localhost:11434"
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    prompt_version: str = "ecg-narrative-v1"
    fallback_enabled: bool = True

    def __post_init__(self) -> None:
        required_values = (
            self.model_name,
            self.host,
            self.prompt_version,
        )

        if any(not value.strip() for value in required_values):
            raise ValueError(
                "model_name, host, and prompt_version cannot be empty"
            )

        if (
            not isfinite(self.temperature)
            or not 0.0 <= self.temperature <= 2.0
        ):
            raise ValueError(
                "temperature must be finite and between zero and two"
            )

        if (
            not isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0.0
        ):
            raise ValueError(
                "timeout_seconds must be finite and greater than zero"
            )
