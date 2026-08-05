from dataclasses import dataclass

from reasoning_pipeline.domain.enums.statuses import SourceDataset


@dataclass(frozen=True)
class ECGSignal:
    record_id: str
    samples: tuple[float, ...]
    sampling_rate_hz: float
    source: str
    lead_name: str | None = None
    source_format: str | None = None
    original_sampling_rate_hz: float | None = None
    lead_names: tuple[str, ...] = ()
    units: str | None = None
    original_sample_count: int | None = None
    original_duration_seconds: float | None = None
    warnings: tuple[str, ...] = ()
    original_units: str | None = None
    target_sampling_rate_hz: float | None = None
    target_units: str | None = None
    resampled: bool = False
    unit_conversion_applied: str | None = None
    resampling_method: str | None = None
    resampling_up_factor: int | None = None
    resampling_down_factor: int | None = None
    harmonised_sample_count: int | None = None
    harmonised_duration_seconds: float | None = None
    harmonisation_transformations: tuple[str, ...] = ()
    harmonisation_warnings: tuple[str, ...] = ()
    source_dataset: SourceDataset | None = None

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise ValueError("record_id cannot be empty")

        if not self.samples:
            raise ValueError("samples cannot be empty")

        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be greater than zero")

        if not self.source.strip():
            raise ValueError("source cannot be empty")

        if self.original_sample_count is not None and self.original_sample_count <= 0:
            raise ValueError("original_sample_count must be greater than zero")

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sampling_rate_hz
