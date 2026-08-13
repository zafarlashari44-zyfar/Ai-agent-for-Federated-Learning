from pathlib import Path

import pytest

from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.infrastructure.input_adapters.npy_adapter import (
    NpyECGInputAdapter,
)


class StubInputService:
    def __init__(self, signal: ECGSignal) -> None:
        self.signal = signal
        self.received_file_path: str | Path | None = None
        self.received_source: str | None = None

    def load_npy(
        self,
        *,
        file_path: str | Path,
        sampling_rate_hz: float,
        record_id: str | None = None,
        source: str = "npy",
        lead_name: str | None = None,
    ) -> ECGSignal:
        self.received_file_path = file_path
        self.received_source = source
        return self.signal


def make_signal() -> ECGSignal:
    return ECGSignal(
        record_id="record-001",
        samples=(0.1, 0.2, 0.3),
        sampling_rate_hz=360.0,
        source="npy",
    )


def test_supports_numpy_suffix_case_insensitively() -> None:
    adapter = NpyECGInputAdapter()

    assert adapter.supports("record.npy")
    assert adapter.supports("record.NPY")
    assert not adapter.supports("record.csv")


def test_load_delegates_to_existing_scribe_input_service() -> None:
    signal = make_signal()
    input_service = StubInputService(signal)

    adapter = NpyECGInputAdapter(
        input_service=input_service,
    )

    returned_signal = adapter.load(
        file_path="record.npy",
        sampling_rate_hz=360.0,
        source="mit-bih",
    )

    assert returned_signal.samples == signal.samples
    assert returned_signal.source_format == "npy"
    assert returned_signal.original_sample_count == signal.sample_count
    assert input_service.received_file_path == Path("record.npy")
    assert input_service.received_source == "mit-bih"


def test_load_uses_default_source() -> None:
    signal = make_signal()
    input_service = StubInputService(signal)

    adapter = NpyECGInputAdapter(
        input_service=input_service,
    )

    adapter.load(
        file_path="record.npy",
        sampling_rate_hz=360.0,
    )

    assert input_service.received_source == "npy"


def test_load_rejects_non_numpy_file() -> None:
    adapter = NpyECGInputAdapter()

    with pytest.raises(
        ValueError,
        match="supports only .npy files",
    ):
        adapter.load(
            file_path="record.csv",
            sampling_rate_hz=360.0,
        )
