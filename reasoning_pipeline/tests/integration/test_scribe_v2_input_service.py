from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reasoning_pipeline.domain.exceptions.pipeline_errors import (
    InvalidSignalError,
    UnsupportedSamplingRateError,
)
from reasoning_pipeline.domain.models import ECGSignal
from reasoning_pipeline.scribe_v2 import ScribeV2InputService


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


def test_service_loads_valid_npy_as_ecg_signal(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "record_001.npy"
    signal = create_valid_signal()

    np.save(file_path, signal)

    service = ScribeV2InputService()

    result = service.load_npy(
        file_path=file_path,
        sampling_rate_hz=100.0,
        source="synthetic_dataset",
        lead_name="Lead II",
    )

    assert isinstance(result, ECGSignal)
    assert result.record_id == "record_001"
    assert result.source == "synthetic_dataset"
    assert result.lead_name == "Lead II"
    assert result.sampling_rate_hz == pytest.approx(100.0)
    assert result.sample_count == 1000
    assert result.duration_seconds == pytest.approx(10.0)
    np.testing.assert_allclose(
        np.asarray(result.samples),
        signal,
    )


def test_service_accepts_custom_record_id(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "record.npy"
    np.save(file_path, create_valid_signal())

    result = ScribeV2InputService().load_npy(
        file_path=file_path,
        sampling_rate_hz=100.0,
        record_id="patient_record_123",
    )

    assert result.record_id == "patient_record_123"


def test_service_strips_record_metadata(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "record.npy"
    np.save(file_path, create_valid_signal())

    result = ScribeV2InputService().load_npy(
        file_path=file_path,
        sampling_rate_hz=100.0,
        record_id="  record_123  ",
        source="  mit_bih  ",
        lead_name="  MLII  ",
    )

    assert result.record_id == "record_123"
    assert result.source == "mit_bih"
    assert result.lead_name == "MLII"


def test_service_rejects_empty_record_id(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "record.npy"
    np.save(file_path, create_valid_signal())

    with pytest.raises(
        ValueError,
        match="record_id cannot be empty",
    ):
        ScribeV2InputService().load_npy(
            file_path=file_path,
            sampling_rate_hz=100.0,
            record_id="   ",
        )


def test_service_rejects_empty_source(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "record.npy"
    np.save(file_path, create_valid_signal())

    with pytest.raises(
        ValueError,
        match="source cannot be empty",
    ):
        ScribeV2InputService().load_npy(
            file_path=file_path,
            sampling_rate_hz=100.0,
            source="   ",
        )


def test_service_propagates_missing_file_error(
    tmp_path: Path,
) -> None:
    missing_file = tmp_path / "missing.npy"

    with pytest.raises(
        FileNotFoundError,
        match="ECG file does not exist",
    ):
        ScribeV2InputService().load_npy(
            file_path=missing_file,
            sampling_rate_hz=100.0,
        )


def test_service_propagates_signal_validation_error(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "constant.npy"

    np.save(
        file_path,
        np.ones(500, dtype=np.float64),
    )

    with pytest.raises(
        InvalidSignalError,
        match="ECG signal is constant",
    ):
        ScribeV2InputService().load_npy(
            file_path=file_path,
            sampling_rate_hz=100.0,
        )


def test_service_propagates_sampling_rate_error(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "record.npy"
    np.save(file_path, create_valid_signal())

    with pytest.raises(
        UnsupportedSamplingRateError,
        match="Unsupported sampling rate",
    ):
        ScribeV2InputService().load_npy(
            file_path=file_path,
            sampling_rate_hz=25.0,
        )


def test_service_output_is_immutable(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "record.npy"
    np.save(file_path, create_valid_signal())

    result = ScribeV2InputService().load_npy(
        file_path=file_path,
        sampling_rate_hz=100.0,
    )

    with pytest.raises(AttributeError):
        result.record_id = "changed"  # type: ignore[misc]