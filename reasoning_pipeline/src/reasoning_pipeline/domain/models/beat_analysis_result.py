from dataclasses import dataclass

from reasoning_pipeline.domain.models.model_prediction import ModelPrediction


@dataclass(frozen=True)
class BeatAnalysisResult:
    """Prediction and source coordinates for one valid ECG beat."""

    beat_index: int
    r_peak_sample_index: int
    source_start_sample_index: int
    source_stop_sample_index_exclusive: int
    r_peak_timestamp_seconds: float
    source_start_timestamp_seconds: float
    source_stop_timestamp_seconds_exclusive: float
    sampling_rate_hz: float
    prediction: ModelPrediction

    def __post_init__(self) -> None:
        if self.beat_index < 0:
            raise ValueError("beat_index cannot be negative")

        if (
            self.source_stop_sample_index_exclusive
            - self.source_start_sample_index
            != 216
        ):
            raise ValueError("Beat analysis source window must contain 216 samples")

        if (
            self.r_peak_sample_index - self.source_start_sample_index
            != 72
        ):
            raise ValueError(
                "Beat analysis R-peak must be 72 samples after window start"
            )
