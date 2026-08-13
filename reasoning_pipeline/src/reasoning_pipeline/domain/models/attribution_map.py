from dataclasses import dataclass
from math import isfinite

from reasoning_pipeline.domain.models.attribution_point import AttributionPoint


@dataclass(frozen=True)
class AttributionMap:
    """One explainer's immutable attribution map for a prepared ECG beat."""

    method_id: str
    method_version: str
    target_class: int
    target_label: str
    target_output: str
    points: tuple[AttributionPoint, ...]
    signed: bool
    native_resolution: int
    interpolation_method: str | None
    normalisation: str
    sampling_rate_hz: float
    source_start_sample_index: int
    source_stop_sample_index_exclusive: int
    convergence_delta: float | None = None
    parameters: tuple[tuple[str, str], ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("method_id", self.method_id),
            ("method_version", self.method_version),
            ("target_label", self.target_label),
            ("target_output", self.target_output),
            ("normalisation", self.normalisation),
        ):
            if not value.strip():
                raise ValueError(f"{name} cannot be empty")

        if not 0 <= self.target_class < 5:
            raise ValueError("target_class must be between 0 and 4")
        if self.native_resolution <= 0:
            raise ValueError("native_resolution must be greater than zero")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be greater than zero")
        if len(self.points) != 216:
            raise ValueError("Attribution map must contain exactly 216 points")
        if (
            self.source_stop_sample_index_exclusive
            - self.source_start_sample_index
            != 216
        ):
            raise ValueError("Attribution source window must contain 216 samples")
        if self.convergence_delta is not None and not isfinite(
            self.convergence_delta
        ):
            raise ValueError("convergence_delta must be finite")

        parameter_names = tuple(name for name, _ in self.parameters)
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("Attribution parameter names must be unique")

        for local_index, point in enumerate(self.points):
            expected_source_index = (
                self.source_start_sample_index + local_index
            )
            if point.beat_sample_index != local_index:
                raise ValueError("Attribution points must use ordered local indices")
            if point.source_sample_index != expected_source_index:
                raise ValueError(
                    "Attribution points must map to consecutive source indices"
                )
            expected_timestamp = (
                expected_source_index / self.sampling_rate_hz
            )
            if not isfinite(expected_timestamp) or abs(
                point.timestamp_seconds - expected_timestamp
            ) > 1e-9:
                raise ValueError(
                    "Attribution timestamps must derive from source indices"
                )
