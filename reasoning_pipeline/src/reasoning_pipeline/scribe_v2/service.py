from __future__ import annotations

from pathlib import Path

from reasoning_pipeline.domain.models import ECGSignal
from reasoning_pipeline.scribe_v2.loader import NpyECGLoader
from reasoning_pipeline.scribe_v2.validator import ECGSignalValidator


class ScribeV2InputService:
    """
    Load and validate ECG data before feature extraction.

    This service is the public entry point for the initial Scribe v2
    input stage. It converts a supported ECG file into the standard
    ECGSignal domain model used by the rest of the reasoning pipeline.
    """

    def __init__(
        self,
        *,
        loader: NpyECGLoader | None = None,
        validator: ECGSignalValidator | None = None,
    ) -> None:
        self.loader = loader or NpyECGLoader()
        self.validator = validator or ECGSignalValidator()

    def load_npy(
        self,
        *,
        file_path: str | Path,
        sampling_rate_hz: float,
        record_id: str | None = None,
        source: str = "npy",
        lead_name: str | None = None,
    ) -> ECGSignal:
        """
        Load a NumPy ECG file and return a validated ECGSignal.

        Args:
            file_path:
                Path to the .npy file.

            sampling_rate_hz:
                Sampling frequency of the ECG in Hertz.

            record_id:
                Optional record identifier. When omitted, the file stem
                is used.

            source:
                Description of the ECG source or dataset.

            lead_name:
                Optional ECG lead name, such as Lead II.

        Returns:
            A validated immutable ECGSignal domain object.
        """
        path = Path(file_path).expanduser()

        resolved_record_id = (
            record_id.strip()
            if record_id is not None
            else path.stem
        )

        if not resolved_record_id:
            raise ValueError("record_id cannot be empty")

        if not source.strip():
            raise ValueError("source cannot be empty")

        loaded_signal = self.loader.load(path)

        validated_signal = self.validator.validate(
            signal=loaded_signal,
            sampling_rate_hz=sampling_rate_hz,
        )

        return ECGSignal(
            record_id=resolved_record_id,
            samples=tuple(float(value) for value in validated_signal),
            sampling_rate_hz=float(sampling_rate_hz),
            source=source.strip(),
            lead_name=lead_name.strip() if lead_name else None,
        )