from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class RecordingAttributionPoint:
    """Aggregated explainability values for one source ECG sample."""

    source_sample_index: int
    timestamp_seconds: float
    maximum_attribution: float
    mean_attribution: float
    coverage_count: int
    contributing_beat_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.source_sample_index < 0:
            raise ValueError("source_sample_index cannot be negative")
        if not isfinite(self.timestamp_seconds):
            raise ValueError("timestamp_seconds must be finite")
        for name, value in (
            ("maximum_attribution", self.maximum_attribution),
            ("mean_attribution", self.mean_attribution),
        ):
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        if self.coverage_count < 0:
            raise ValueError("coverage_count cannot be negative")
        if self.coverage_count != len(self.contributing_beat_indices):
            raise ValueError(
                "coverage_count must match contributing_beat_indices"
            )
        if self.contributing_beat_indices != tuple(
            sorted(set(self.contributing_beat_indices))
        ):
            raise ValueError(
                "contributing beat indices must be unique and ordered"
            )
        if self.coverage_count == 0 and (
            self.maximum_attribution != 0.0
            or self.mean_attribution != 0.0
        ):
            raise ValueError("Uncovered samples must have zero attribution")
