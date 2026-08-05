from __future__ import annotations

from pathlib import Path

import pandas as pd

from reasoning_pipeline.domain.enums.statuses import SourceDataset
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.infrastructure.input_adapters.common import (
    build_signal,
    require_sampling_rate,
)


class CsvECGInputAdapter:
    supported_suffixes = (".csv",)

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
        del companion_file_path
        path = Path(file_path)
        try:
            frame = pd.read_csv(path)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            raise ValueError("Unable to read ECG CSV file.") from exc
        if frame.empty:
            raise ValueError("ECG signal cannot be empty.")
        numeric_columns = tuple(
            str(name)
            for name in frame.columns
            if pd.api.types.is_numeric_dtype(frame[name])
        )
        if signal_column is not None:
            if signal_column not in frame.columns:
                raise ValueError(f"CSV signal column '{signal_column}' was not found.")
            column = signal_column
            if not pd.api.types.is_numeric_dtype(frame[column]):
                raise ValueError(
                    f"CSV signal column '{column}' must contain numeric values."
                )
        elif len(numeric_columns) == 1:
            column = numeric_columns[0]
        elif not numeric_columns:
            raise ValueError("CSV does not contain a numeric signal column.")
        else:
            raise ValueError(
                "CSV contains multiple numeric columns; signal_column must "
                "be specified."
            )
        return build_signal(
            path=path,
            values=frame[column].to_numpy(),
            sampling_rate_hz=require_sampling_rate(sampling_rate_hz),
            source_format="csv",
            record_id=record_id,
            source=source,
            selected_lead=lead_name,
            lead_names=(lead_name,) if lead_name else (),
            units=units,
            source_dataset=source_dataset,
        )
