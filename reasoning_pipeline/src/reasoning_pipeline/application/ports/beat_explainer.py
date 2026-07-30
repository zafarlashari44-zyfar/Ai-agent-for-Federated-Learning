from dataclasses import dataclass
from math import isfinite
from typing import Protocol


@dataclass(frozen=True)
class LocalAttribution:
    """Explainer output in the 216-sample prepared-beat coordinate system."""

    method_id: str
    method_version: str
    target_class: int
    target_output: str
    values: tuple[float, ...]
    signed: bool
    native_resolution: int
    interpolation_method: str | None
    normalisation: str
    convergence_delta: float | None = None
    parameters: tuple[tuple[str, str], ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("method_id", self.method_id),
            ("method_version", self.method_version),
            ("target_output", self.target_output),
            ("normalisation", self.normalisation),
        ):
            if not value.strip():
                raise ValueError(f"{name} cannot be empty")
        if not 0 <= self.target_class < 5:
            raise ValueError("target_class must be between 0 and 4")
        if len(self.values) != 216:
            raise ValueError("Local attribution must contain exactly 216 values")
        if not all(isfinite(value) for value in self.values):
            raise ValueError("Local attribution values must be finite")
        if self.native_resolution <= 0:
            raise ValueError("native_resolution must be greater than zero")
        if self.convergence_delta is not None and not isfinite(
            self.convergence_delta
        ):
            raise ValueError("convergence_delta must be finite")


class BeatExplainerProtocol(Protocol):
    @property
    def method_id(self) -> str:
        ...

    def explain(
        self,
        *,
        samples: tuple[float, ...],
        target_class: int,
    ) -> LocalAttribution | None:
        ...
