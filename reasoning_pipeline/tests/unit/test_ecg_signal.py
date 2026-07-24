from __future__ import annotations

import pytest

from reasoning_pipeline.domain.models import ECGSignal


def test_ecg_signal_is_created_with_valid_data() -> None:
    signal = ECGSignal(
        record_id="record_001",
        samples=(0.1, 0.2, 0.3, 0.4),
        sampling_rate_hz=2.0,
        source="synthetic",
        lead_name="Lead II",
    )

    assert signal.record_id == "record_001"
    assert signal.sample_count == 4
    assert signal.duration_seconds == pytest.approx(2.0)
    assert signal.lead_name == "Lead II"


def test_ecg_signal_rejects_empty_record_id() -> None:
    with pytest.raises(ValueError, match="record_id cannot be empty"):
        ECGSignal(
            record_id="",
            samples=(0.1, 0.2),
            sampling_rate_hz=100.0,
            source="synthetic",
        )


def test_ecg_signal_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="samples cannot be empty"):
        ECGSignal(
            record_id="record_001",
            samples=(),
            sampling_rate_hz=100.0,
            source="synthetic",
        )


def test_ecg_signal_rejects_invalid_sampling_rate() -> None:
    with pytest.raises(
        ValueError,
        match="sampling_rate_hz must be greater than zero",
    ):
        ECGSignal(
            record_id="record_001",
            samples=(0.1, 0.2),
            sampling_rate_hz=0.0,
            source="synthetic",
        )


def test_ecg_signal_rejects_empty_source() -> None:
    with pytest.raises(ValueError, match="source cannot be empty"):
        ECGSignal(
            record_id="record_001",
            samples=(0.1, 0.2),
            sampling_rate_hz=100.0,
            source="",
        )
