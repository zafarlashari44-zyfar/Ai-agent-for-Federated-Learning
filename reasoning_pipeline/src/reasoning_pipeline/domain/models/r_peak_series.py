from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class RPeakSeries:
    sample_indices: Tuple[int, ...]
    timestamps_seconds: Tuple[float, ...]
    rr_intervals_ms: Tuple[float, ...]
    detector_name: str
    detector_version: str
    confidence: float

    def __post_init__(self) -> None:
        if len(self.sample_indices) != len(self.timestamps_seconds):
            raise ValueError(
                "sample_indices and timestamps_seconds must match"
            )

        if any(index < 0 for index in self.sample_indices):
            raise ValueError("R peak sample indices cannot be negative")

        if any(interval <= 0 for interval in self.rr_intervals_ms):
            raise ValueError("RR intervals must be greater than zero")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")

    @property
    def peak_count(self) -> int:
        return len(self.sample_indices)
