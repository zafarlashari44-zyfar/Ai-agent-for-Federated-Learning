from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class AttributionPoint:
    """One model-input attribution mapped to the source ECG recording."""

    beat_sample_index: int
    source_sample_index: int
    timestamp_seconds: float
    attribution: float
    input_value: float

    def __post_init__(self) -> None:
        if not 0 <= self.beat_sample_index < 216:
            raise ValueError("beat_sample_index must be between 0 and 215")
        if self.source_sample_index < 0:
            raise ValueError("source_sample_index cannot be negative")
        if not all(
            isfinite(value)
            for value in (
                self.timestamp_seconds,
                self.attribution,
                self.input_value,
            )
        ):
            raise ValueError("Attribution point values must be finite")
