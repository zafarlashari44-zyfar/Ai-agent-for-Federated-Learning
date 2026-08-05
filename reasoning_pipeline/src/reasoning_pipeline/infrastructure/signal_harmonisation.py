from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

from reasoning_pipeline.domain.models.ecg_signal import ECGSignal


class ScipySignalHarmoniser:
    """Harmonise amplitude and sampling rate for the frozen ECG model."""

    TARGET_UNITS = "mV"
    RESAMPLING_METHOD = "scipy.signal.resample_poly"
    RATE_INTEGER_TOLERANCE = 1e-9
    DURATION_EPSILON_SECONDS = 1e-12

    def __init__(
        self,
        *,
        target_sampling_rate_hz: float,
        allow_legacy_npy_missing_units: bool = True,
    ) -> None:
        self._validate_rate(target_sampling_rate_hz, name="target sampling rate")
        self.target_sampling_rate_hz = float(target_sampling_rate_hz)
        self.allow_legacy_npy_missing_units = allow_legacy_npy_missing_units

    @property
    def duration_tolerance_seconds(self) -> float:
        return (
            1.0 / self.target_sampling_rate_hz
            + self.DURATION_EPSILON_SECONDS
        )

    def harmonise(self, signal: ECGSignal) -> ECGSignal:
        values = np.asarray(signal.samples, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("Selected ECG signal must be one-dimensional.")
        if values.size == 0:
            raise ValueError("Selected ECG signal cannot be empty.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Selected ECG signal contains NaN or infinite values.")
        self._validate_rate(signal.sampling_rate_hz, name="source sampling rate")

        original_units, scale, conversion, unit_warnings = self._unit_conversion(
            signal
        )
        harmonised = values * scale
        transformations = [conversion]
        source_rate = float(signal.sampling_rate_hz)
        resampled = not np.isclose(
            source_rate,
            self.target_sampling_rate_hz,
            rtol=0.0,
            atol=self.RATE_INTEGER_TOLERANCE,
        )
        up_factor = 1
        down_factor = 1
        method: str | None = None
        if resampled:
            source_integer = self._integral_rate(source_rate, "source sampling rate")
            target_integer = self._integral_rate(
                self.target_sampling_rate_hz,
                "target sampling rate",
            )
            divisor = gcd(source_integer, target_integer)
            up_factor = target_integer // divisor
            down_factor = source_integer // divisor
            if up_factor <= 0 or down_factor <= 0:
                raise ValueError("Unsafe resampling factor configuration.")
            harmonised = resample_poly(harmonised, up_factor, down_factor)
            method = self.RESAMPLING_METHOD
            transformations.append(
                f"resampled {source_rate:g} Hz to "
                f"{self.target_sampling_rate_hz:g} Hz using {method} "
                f"(up={up_factor}, down={down_factor})"
            )
        else:
            transformations.append("sampling rate already matches model contract")

        if harmonised.size == 0:
            raise ValueError("Harmonised ECG signal cannot be empty.")
        if not np.all(np.isfinite(harmonised)):
            raise ValueError("Harmonised ECG signal contains non-finite values.")

        source_duration = values.size / source_rate
        harmonised_duration = harmonised.size / self.target_sampling_rate_hz
        duration_difference = abs(harmonised_duration - source_duration)
        if duration_difference > self.duration_tolerance_seconds:
            raise ValueError(
                "Resampling did not preserve signal duration within the "
                f"{self.duration_tolerance_seconds:.12f} second tolerance."
            )

        return ECGSignal(
            record_id=signal.record_id,
            samples=tuple(float(value) for value in harmonised),
            sampling_rate_hz=self.target_sampling_rate_hz,
            source=signal.source,
            lead_name=signal.lead_name,
            source_format=signal.source_format,
            original_sampling_rate_hz=(
                signal.original_sampling_rate_hz or source_rate
            ),
            lead_names=signal.lead_names,
            units=self.TARGET_UNITS,
            original_sample_count=signal.original_sample_count or values.size,
            original_duration_seconds=(
                signal.original_duration_seconds or source_duration
            ),
            warnings=signal.warnings,
            original_units=original_units,
            target_sampling_rate_hz=self.target_sampling_rate_hz,
            target_units=self.TARGET_UNITS,
            resampled=resampled,
            unit_conversion_applied=conversion,
            resampling_method=method,
            resampling_up_factor=up_factor,
            resampling_down_factor=down_factor,
            harmonised_sample_count=int(harmonised.size),
            harmonised_duration_seconds=harmonised_duration,
            harmonisation_transformations=tuple(transformations),
            harmonisation_warnings=unit_warnings,
            source_dataset=signal.source_dataset,
        )

    def _unit_conversion(
        self,
        signal: ECGSignal,
    ) -> tuple[str | None, float, str, tuple[str, ...]]:
        units = signal.units
        if units is None or not units.strip():
            if (
                self.allow_legacy_npy_missing_units
                and signal.source_format == "npy"
            ):
                warning = (
                    "Legacy NPY compatibility mode used: missing amplitude "
                    "units were treated as mV."
                )
                return None, 1.0, "compatibility assumption: mV", (warning,)
            raise ValueError(
                "ECG amplitude units are required; supported units are "
                "mV, uV, µV, and V."
            )

        normalized = units.strip().replace("μ", "µ")
        casefolded = normalized.casefold()
        if casefolded == "mv":
            return units, 1.0, "mV to mV (scale=1)", ()
        if normalized in {"uV", "µV"} or casefolded in {"uv", "µv"}:
            return units, 0.001, f"{units} to mV (scale=0.001)", ()
        if normalized == "V" or casefolded == "v":
            return units, 1000.0, f"{units} to mV (scale=1000)", ()
        raise ValueError(
            f"Unsupported or ambiguous ECG amplitude units '{units}'. "
            "Supported units are mV, uV, µV, and V."
        )

    def _integral_rate(self, value: float, name: str) -> int:
        rounded = round(value)
        if not np.isclose(
            value,
            rounded,
            rtol=0.0,
            atol=self.RATE_INTEGER_TOLERANCE,
        ):
            raise ValueError(
                f"Unsafe resampling configuration: {name} must be integral; "
                f"received {value:g} Hz."
            )
        return int(rounded)

    @staticmethod
    def _validate_rate(value: float, *, name: str) -> None:
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"{name.capitalize()} must be numeric.")
        if not np.isfinite(value) or float(value) <= 0:
            raise ValueError(f"{name.capitalize()} must be finite and positive.")
