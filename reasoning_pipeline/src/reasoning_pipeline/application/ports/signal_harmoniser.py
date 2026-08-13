from typing import Protocol

from reasoning_pipeline.domain.models.ecg_signal import ECGSignal


class SignalHarmoniserProtocol(Protocol):
    """Convert an ingested single-lead ECG to the frozen model contract."""

    def harmonise(self, signal: ECGSignal) -> ECGSignal:
        ...
