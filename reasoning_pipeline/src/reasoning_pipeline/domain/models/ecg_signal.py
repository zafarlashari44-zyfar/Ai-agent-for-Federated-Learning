from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ECGSignal:
    record_id: str
    samples: Tuple[float, ...]
    sampling_rate_hz: float
    source: str
    lead_name: str | None = None

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise ValueError("record_id cannot be empty")

        if not self.samples:
            raise ValueError("samples cannot be empty")

        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be greater than zero")

        if not self.source.strip():
            raise ValueError("source cannot be empty")

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sampling_rate_hz
