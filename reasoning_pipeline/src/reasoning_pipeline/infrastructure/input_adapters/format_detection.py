from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from reasoning_pipeline.application.ports.ecg_input_adapter import (
    ECGInputAdapterProtocol,
)


class ECGFormatDetectionService:
    def __init__(self, adapters: Sequence[ECGInputAdapterProtocol]) -> None:
        self._adapters = tuple(adapters)

    def detect(self, file_path: str | Path) -> ECGInputAdapterProtocol:
        for adapter in self._adapters:
            if adapter.supports(file_path):
                return adapter
        path = Path(file_path)
        supported = sorted(
            {
                suffix
                for adapter in self._adapters
                for suffix in adapter.supported_suffixes
            }
        )
        raise ValueError(
            f"Unsupported ECG file format {path.suffix or '<no extension>'}. "
            f"Supported formats are {', '.join(supported)}."
        )
