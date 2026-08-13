from dataclasses import dataclass

from reasoning_pipeline.domain.models.attribution_map import AttributionMap


@dataclass(frozen=True)
class BeatExplanation:
    """All available explanation methods for one predicted ECG beat."""

    beat_index: int
    r_peak_sample_index: int
    r_peak_timestamp_seconds: float
    source_start_sample_index: int
    source_stop_sample_index_exclusive: int
    source_start_timestamp_seconds: float
    source_stop_timestamp_seconds_exclusive: float
    sampling_rate_hz: float
    predicted_class: int
    predicted_label: str
    prediction_confidence: float
    attribution_maps: tuple[AttributionMap, ...]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.beat_index < 0:
            raise ValueError("beat_index cannot be negative")
        if (
            self.source_stop_sample_index_exclusive
            - self.source_start_sample_index
            != 216
        ):
            raise ValueError("Beat explanation source window must contain 216 samples")
        if self.r_peak_sample_index - self.source_start_sample_index != 72:
            raise ValueError(
                "Beat explanation R-peak must be 72 samples after window start"
            )
        if not 0 <= self.predicted_class < 5:
            raise ValueError("predicted_class must be between 0 and 4")
        if not self.predicted_label.strip():
            raise ValueError("predicted_label cannot be empty")
        if not 0.0 <= self.prediction_confidence <= 1.0:
            raise ValueError("prediction_confidence must be between zero and one")

        method_ids = tuple(
            attribution.method_id
            for attribution in self.attribution_maps
        )
        if len(set(method_ids)) != len(method_ids):
            raise ValueError("Attribution method IDs must be unique per beat")

        for attribution in self.attribution_maps:
            if (
                attribution.source_start_sample_index
                != self.source_start_sample_index
                or attribution.source_stop_sample_index_exclusive
                != self.source_stop_sample_index_exclusive
            ):
                raise ValueError(
                    "Attribution maps must use the beat source window"
                )
