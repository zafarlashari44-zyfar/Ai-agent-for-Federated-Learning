from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.orchestration.baseline_signal_cleaner import (
    FrozenBaselineSignalCleaner,
)


def _signal() -> ECGSignal:
    return ECGSignal(
        record_id="record-001",
        samples=tuple(float(value) for value in range(500)),
        sampling_rate_hz=360.0,
        source="unit-test",
        lead_name="MLII",
    )


def test_cleaner_uses_neurokit_ecg_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal = _signal()
    captured: dict[str, object] = {}

    cleaned = np.linspace(
        -1.0,
        1.0,
        len(signal.samples),
    )

    def fake_ecg_process(
        samples,
        *,
        sampling_rate,
        method,
    ):
        captured["samples"] = np.asarray(samples)
        captured["sampling_rate"] = sampling_rate
        captured["method"] = method

        return (
            pd.DataFrame(
                {
                    "ECG_Clean": cleaned,
                }
            ),
            {},
        )

    monkeypatch.setattr(
        "reasoning_pipeline.orchestration."
        "baseline_signal_cleaner.nk.ecg_process",
        fake_ecg_process,
    )

    cleaner = FrozenBaselineSignalCleaner()
    result = cleaner.clean(signal)

    assert captured["sampling_rate"] == 360
    assert captured["method"] == "neurokit"
    assert np.array_equal(
        captured["samples"],
        np.asarray(signal.samples),
    )
    assert np.array_equal(result, cleaned)


def test_cleaner_rejects_missing_ecg_clean_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_ecg_process(
        samples,
        *,
        sampling_rate,
        method,
    ):
        return (
            pd.DataFrame(
                {
                    "ECG_Raw": samples,
                }
            ),
            {},
        )

    monkeypatch.setattr(
        "reasoning_pipeline.orchestration."
        "baseline_signal_cleaner.nk.ecg_process",
        fake_ecg_process,
    )

    with pytest.raises(
        RuntimeError,
        match="did not return ECG_Clean",
    ):
        FrozenBaselineSignalCleaner().clean(_signal())


def test_cleaner_rejects_changed_signal_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_ecg_process(
        samples,
        *,
        sampling_rate,
        method,
    ):
        return (
            pd.DataFrame(
                {
                    "ECG_Clean": np.zeros(
                        len(samples) - 1,
                    ),
                }
            ),
            {},
        )

    monkeypatch.setattr(
        "reasoning_pipeline.orchestration."
        "baseline_signal_cleaner.nk.ecg_process",
        fake_ecg_process,
    )

    with pytest.raises(
        RuntimeError,
        match="length does not match",
    ):
        FrozenBaselineSignalCleaner().clean(_signal())
