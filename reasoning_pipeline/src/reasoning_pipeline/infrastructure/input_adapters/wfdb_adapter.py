from __future__ import annotations

from pathlib import Path

import numpy as np
import wfdb

from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.infrastructure.input_adapters.common import build_signal


class WfdbECGInputAdapter:
    supported_suffixes = (".hea", ".dat")

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
    ) -> ECGSignal:
        del sampling_rate_hz, signal_column, units
        first = Path(file_path)
        companion = Path(companion_file_path) if companion_file_path else None
        paths = {first.suffix.lower(): first}
        if companion is not None:
            paths[companion.suffix.lower()] = companion
        if ".hea" not in paths:
            raise ValueError("WFDB upload is missing the .hea header file.")
        if ".dat" not in paths:
            raise ValueError("WFDB upload is missing the .dat data file.")
        header = paths[".hea"]
        data = paths[".dat"]
        if header.stem != data.stem:
            raise ValueError("WFDB .hea and .dat files must have the same record name.")
        try:
            record = wfdb.rdrecord(str(header.with_suffix("")))
        except (OSError, ValueError, IndexError) as exc:
            raise ValueError("Unable to read the WFDB record pair.") from exc
        lead_names = tuple(str(name) for name in record.sig_name)
        if not lead_names:
            raise ValueError("WFDB record does not contain any leads.")
        warning: tuple[str, ...] = ()
        if lead_name is not None:
            if lead_name not in lead_names:
                raise ValueError(
                    f"WFDB lead '{lead_name}' was not found. Available leads: "
                    + ", ".join(lead_names)
                )
            selected = lead_name
        elif "II" in lead_names:
            selected = "II"
        elif "Lead II" in lead_names:
            selected = "Lead II"
        elif "MLII" in lead_names:
            selected = "MLII"
        else:
            selected = lead_names[0]
            warning = (f"No Lead II or MLII found; selected first lead '{selected}'.",)
        index = lead_names.index(selected)
        values = np.asarray(record.p_signal[:, index])
        record_units = tuple(str(value) for value in record.units)
        return build_signal(
            path=header,
            values=values,
            sampling_rate_hz=float(record.fs),
            source_format="wfdb",
            record_id=record_id,
            source=source,
            selected_lead=selected,
            lead_names=lead_names,
            units=record_units[index] if record_units else None,
            warnings=warning,
        )
