from dataclasses import dataclass

from reasoning_pipeline.domain.models.recording_attribution_point import (
    RecordingAttributionPoint,
)


@dataclass(frozen=True)
class RecordingAttributionOverlay:
    """Full-record attribution suitable for waveform heatmap rendering."""

    record_id: str
    method_id: str
    method_version: str
    sampling_rate_hz: float
    total_source_samples: int
    aggregation_method: str
    points: tuple[RecordingAttributionPoint, ...]
    explained_beat_count: int
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("record_id", self.record_id),
            ("method_id", self.method_id),
            ("method_version", self.method_version),
            ("aggregation_method", self.aggregation_method),
        ):
            if not value.strip():
                raise ValueError(f"{name} cannot be empty")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be greater than zero")
        if self.total_source_samples <= 0:
            raise ValueError("total_source_samples must be greater than zero")
        if len(self.points) != self.total_source_samples:
            raise ValueError(
                "Overlay must contain exactly one point per source sample"
            )
        if self.explained_beat_count < 0:
            raise ValueError("explained_beat_count cannot be negative")

        for source_index, point in enumerate(self.points):
            if point.source_sample_index != source_index:
                raise ValueError(
                    "Overlay source indices must be ordered from zero"
                )
            expected_timestamp = source_index / self.sampling_rate_hz
            if abs(point.timestamp_seconds - expected_timestamp) > 1e-9:
                raise ValueError(
                    "Overlay timestamps must derive from source indices"
                )
