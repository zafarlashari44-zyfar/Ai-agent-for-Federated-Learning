from __future__ import annotations

from pathlib import Path

import numpy as np

from reasoning_pipeline.domain.enums.statuses import SourceDataset
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.infrastructure.input_adapters.common import (
    build_signal,
    require_sampling_rate,
)


class TextECGInputAdapter:
    supported_suffixes = (".txt",)

    def supports(self, file_path: str | Path) -> bool:
        return Path(file_path).suffix.lower() in self.supported_suffixes

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
        source_dataset: SourceDataset | None = None,
    ) -> ECGSignal:
        del signal_column, companion_file_path
        path = Path(file_path)
        try:
            values = np.loadtxt(path, dtype=np.float64)
        except (OSError, ValueError) as exc:
            raise ValueError(
                "Text ECG signal must contain only numeric values."
            ) from exc
        return build_signal(
            path=path,
            values=np.asarray(values),
            sampling_rate_hz=require_sampling_rate(sampling_rate_hz),
            source_format="txt",
            record_id=record_id,
            source=source,
            selected_lead=lead_name,
            lead_names=(lead_name,) if lead_name else (),
            units=units,
            source_dataset=source_dataset,
        )
