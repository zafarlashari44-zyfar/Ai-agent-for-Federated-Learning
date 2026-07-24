from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reasoning_pipeline.domain.exceptions.pipeline_errors import (
    InvalidSignalError,
    UnsupportedSamplingRateError,
)

FloatArray = NDArray[np.float64]


class ECGSignalValidator:
    """Validate and normalize a single-lead ECG NumPy array."""

    def __init__(
        self,
        *,
        minimum_duration_seconds: float = 2.0,
        minimum_sampling_rate_hz: float = 50.0,
        maximum_sampling_rate_hz: float = 5000.0,
    ) -> None:
        if minimum_duration_seconds <= 0:
            raise ValueError(
                "minimum_duration_seconds must be greater than zero"
            )

        if minimum_sampling_rate_hz <= 0:
            raise ValueError(
                "minimum_sampling_rate_hz must be greater than zero"
            )

        if maximum_sampling_rate_hz <= minimum_sampling_rate_hz:
            raise ValueError(
                "maximum_sampling_rate_hz must be greater than "
                "minimum_sampling_rate_hz"
            )

        self.minimum_duration_seconds = minimum_duration_seconds
        self.minimum_sampling_rate_hz = minimum_sampling_rate_hz
        self.maximum_sampling_rate_hz = maximum_sampling_rate_hz

    def validate(
        self,
        signal: NDArray[np.generic],
        sampling_rate_hz: float,
    ) -> FloatArray:
        """
        Validate a single-lead ECG signal.

        Args:
            signal:
                NumPy array containing ECG samples.

            sampling_rate_hz:
                Sampling frequency in Hertz.

        Returns:
            A validated contiguous float64 one-dimensional array.

        Raises:
            InvalidSignalError:
                If the signal is empty, multidimensional, non-finite,
                too short, non-numeric, or constant.

            UnsupportedSamplingRateError:
                If the sampling rate is outside the supported range.
        """
        self._validate_sampling_rate(sampling_rate_hz)

        if not isinstance(signal, np.ndarray):
            raise InvalidSignalError(
                "ECG signal must be provided as a NumPy array."
            )

        if signal.ndim != 1:
            raise InvalidSignalError(
                "ECG signal must be one-dimensional. "
                f"Received shape: {signal.shape}"
            )

        if signal.size == 0:
            raise InvalidSignalError("ECG signal cannot be empty.")

        try:
            normalized = np.asarray(
                signal,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise InvalidSignalError(
                "ECG signal must contain numeric values."
            ) from exc

        if not np.all(np.isfinite(normalized)):
            raise InvalidSignalError(
                "ECG signal contains NaN or infinite values."
            )

        minimum_sample_count = int(
            np.ceil(
                self.minimum_duration_seconds
                * sampling_rate_hz
            )
        )

        if normalized.size < minimum_sample_count:
            actual_duration = normalized.size / sampling_rate_hz

            raise InvalidSignalError(
                "ECG signal is too short. "
                f"Minimum duration is "
                f"{self.minimum_duration_seconds:.2f} seconds; "
                f"received {actual_duration:.2f} seconds."
            )

        if np.ptp(normalized) == 0.0:
            raise InvalidSignalError(
                "ECG signal is constant and contains no measurable variation."
            )

        return np.ascontiguousarray(
            normalized,
            dtype=np.float64,
        )

    def _validate_sampling_rate(
        self,
        sampling_rate_hz: float,
    ) -> None:
        if not isinstance(
            sampling_rate_hz,
            (int, float, np.integer, np.floating),
        ):
            raise UnsupportedSamplingRateError(
                "Sampling rate must be numeric."
            )

        if not np.isfinite(sampling_rate_hz):
            raise UnsupportedSamplingRateError(
                "Sampling rate must be finite."
            )

        if not (
            self.minimum_sampling_rate_hz
            <= float(sampling_rate_hz)
            <= self.maximum_sampling_rate_hz
        ):
            raise UnsupportedSamplingRateError(
                "Unsupported sampling rate. "
                f"Expected a value between "
                f"{self.minimum_sampling_rate_hz:.1f} Hz and "
                f"{self.maximum_sampling_rate_hz:.1f} Hz; "
                f"received {float(sampling_rate_hz):.1f} Hz."
            )