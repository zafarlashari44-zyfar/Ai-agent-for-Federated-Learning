from __future__ import annotations

import numpy as np
import pytest

from reasoning_pipeline.domain.exceptions.pipeline_errors import (
    InvalidSignalError,
    UnsupportedSamplingRateError,
)
from reasoning_pipeline.scribe_v2 import ECGSignalValidator


def create_valid_signal(
    sample_count: int = 1000,
) -> np.ndarray:
    time = np.linspace(
        0.0,
        10.0,
        sample_count,
        endpoint=False,
    )

    return np.sin(2.0 * np.pi * 1.2 * time)


def test_validator_accepts_valid_signal() -> None:
    validator = ECGSignalValidator()
    signal = create_valid_signal()

    result = validator.validate(
        signal=signal,
        sampling_rate_hz=100.0,
    )

    assert result.dtype == np.float64
    assert result.ndim == 1
    assert result.flags.c_contiguous
    np.testing.assert_allclose(result, signal)


def test_validator_rejects_non_numpy_input() -> None:
    validator = ECGSignalValidator()

    with pytest.raises(
        InvalidSignalError,
        match="must be provided as a NumPy array",
    ):
        validator.validate(
            signal=[0.1, 0.2, 0.3],  # type: ignore[arg-type]
            sampling_rate_hz=100.0,
        )


def test_validator_rejects_empty_signal() -> None:
    validator = ECGSignalValidator()

    with pytest.raises(
        InvalidSignalError,
        match="cannot be empty",
    ):
        validator.validate(
            signal=np.array([], dtype=np.float64),
            sampling_rate_hz=100.0,
        )


def test_validator_rejects_two_dimensional_signal() -> None:
    validator = ECGSignalValidator()

    signal = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ],
        dtype=np.float64,
    )

    with pytest.raises(
        InvalidSignalError,
        match="must be one-dimensional",
    ):
        validator.validate(
            signal=signal,
            sampling_rate_hz=100.0,
        )


def test_validator_rejects_nan_values() -> None:
    validator = ECGSignalValidator()

    signal = create_valid_signal()
    signal[10] = np.nan

    with pytest.raises(
        InvalidSignalError,
        match="contains NaN or infinite values",
    ):
        validator.validate(
            signal=signal,
            sampling_rate_hz=100.0,
        )


def test_validator_rejects_infinite_values() -> None:
    validator = ECGSignalValidator()

    signal = create_valid_signal()
    signal[10] = np.inf

    with pytest.raises(
        InvalidSignalError,
        match="contains NaN or infinite values",
    ):
        validator.validate(
            signal=signal,
            sampling_rate_hz=100.0,
        )


def test_validator_rejects_signal_that_is_too_short() -> None:
    validator = ECGSignalValidator(
        minimum_duration_seconds=2.0,
    )

    signal = np.array(
        [0.1, 0.2, 0.3],
        dtype=np.float64,
    )

    with pytest.raises(
        InvalidSignalError,
        match="ECG signal is too short",
    ):
        validator.validate(
            signal=signal,
            sampling_rate_hz=100.0,
        )


def test_validator_rejects_constant_signal() -> None:
    validator = ECGSignalValidator()

    signal = np.ones(
        500,
        dtype=np.float64,
    )

    with pytest.raises(
        InvalidSignalError,
        match="ECG signal is constant",
    ):
        validator.validate(
            signal=signal,
            sampling_rate_hz=100.0,
        )


def test_validator_rejects_sampling_rate_below_limit() -> None:
    validator = ECGSignalValidator(
        minimum_sampling_rate_hz=50.0,
    )

    with pytest.raises(
        UnsupportedSamplingRateError,
        match="Unsupported sampling rate",
    ):
        validator.validate(
            signal=create_valid_signal(),
            sampling_rate_hz=25.0,
        )


def test_validator_rejects_sampling_rate_above_limit() -> None:
    validator = ECGSignalValidator(
        maximum_sampling_rate_hz=1000.0,
    )

    with pytest.raises(
        UnsupportedSamplingRateError,
        match="Unsupported sampling rate",
    ):
        validator.validate(
            signal=create_valid_signal(),
            sampling_rate_hz=2000.0,
        )


def test_validator_rejects_non_finite_sampling_rate() -> None:
    validator = ECGSignalValidator()

    with pytest.raises(
        UnsupportedSamplingRateError,
        match="Sampling rate must be finite",
    ):
        validator.validate(
            signal=create_valid_signal(),
            sampling_rate_hz=float("nan"),
        )


def test_validator_rejects_invalid_configuration() -> None:
    with pytest.raises(
        ValueError,
        match="minimum_duration_seconds",
    ):
        ECGSignalValidator(
            minimum_duration_seconds=0.0,
        )

    with pytest.raises(
        ValueError,
        match="maximum_sampling_rate_hz",
    ):
        ECGSignalValidator(
            minimum_sampling_rate_hz=500.0,
            maximum_sampling_rate_hz=100.0,
        )