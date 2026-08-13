from __future__ import annotations

from pathlib import Path

from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.scribe_v2.service import ScribeV2InputService


class NpyECGInputAdapter:
    """
    Convert a NumPy ECG file into a validated ECGSignal.

    This adapter delegates loading and validation to the existing Scribe v2
    input service. It does not duplicate any Phase 1 preprocessing logic.
    """

    DEFAULT_SOURCE = "npy"

    def __init__(
        self,
        *,
        input_service: ScribeV2InputService | None = None,
    ) -> None:
        self._input_service = input_service or ScribeV2InputService()

    @property
    def supported_suffixes(self) -> tuple[str, ...]:
        return (".npy",)

    def supports(self, file_path: str | Path) -> bool:
        path = Path(file_path).expanduser()
        return path.suffix.lower() in self.supported_suffixes

    def load(
        self,
        *,
        file_path: str | Path,
        sampling_rate_hz: float | None = None,
        record_id: str | None = None,
        source: str | None = None,
        lead_name: str | None = None,
        signal_column: str | None = None,
        units: str | None = None,
        companion_file_path: str | Path | None = None,
    ) -> ECGSignal:
        del signal_column, companion_file_path
        path = Path(file_path).expanduser()

        if not self.supports(path):
            raise ValueError(
                "The NumPy ECG adapter supports only .npy files. "
                f"Received {path.suffix or '<no extension>'}."
            )

        resolved_source = (
            source.strip()
            if source is not None
            else self.DEFAULT_SOURCE
        )

        if not resolved_source:
            raise ValueError("source cannot be empty")

        if sampling_rate_hz is None:
            raise ValueError("sampling_rate_hz is required for this file format.")
        signal = self._input_service.load_npy(
            file_path=path,
            sampling_rate_hz=sampling_rate_hz,
            record_id=record_id,
            source=resolved_source,
            lead_name=lead_name,
        )
        return ECGSignal(
            record_id=signal.record_id,
            samples=signal.samples,
            sampling_rate_hz=signal.sampling_rate_hz,
            source=signal.source,
            lead_name=signal.lead_name,
            source_format="npy",
            original_sampling_rate_hz=signal.sampling_rate_hz,
            lead_names=(signal.lead_name,) if signal.lead_name else (),
            units=units.strip() if units else None,
            original_sample_count=signal.sample_count,
            original_duration_seconds=signal.duration_seconds,
        )
