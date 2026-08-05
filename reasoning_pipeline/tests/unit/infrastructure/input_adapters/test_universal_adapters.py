from pathlib import Path

import numpy as np
import pytest
import wfdb

from reasoning_pipeline.domain.enums.statuses import SourceDataset
from reasoning_pipeline.domain.exceptions.pipeline_errors import (
    InvalidSignalError,
)
from reasoning_pipeline.infrastructure.input_adapters.csv_adapter import (
    CsvECGInputAdapter,
)
from reasoning_pipeline.infrastructure.input_adapters.npy_adapter import (
    NpyECGInputAdapter,
)
from reasoning_pipeline.infrastructure.input_adapters.text_adapter import (
    TextECGInputAdapter,
)
from reasoning_pipeline.infrastructure.input_adapters.wfdb_adapter import (
    WfdbECGInputAdapter,
)


def waveform() -> np.ndarray:
    return np.sin(np.linspace(0, 20, 500))


def test_valid_npy_preserves_source_metadata(tmp_path: Path) -> None:
    path = tmp_path / "record.npy"
    np.save(path, waveform())
    signal = NpyECGInputAdapter().load(
        file_path=path,
        sampling_rate_hz=100.0,
        lead_name="II",
        units="mV",
    )
    assert signal.source_format == "npy"
    assert signal.original_sampling_rate_hz == 100.0
    assert signal.original_sample_count == 500
    assert signal.original_duration_seconds == 5.0
    assert signal.lead_names == ("II",)
    assert signal.units == "mV"


def test_single_numeric_csv_column_is_selected(tmp_path: Path) -> None:
    path = tmp_path / "record.csv"
    path.write_text("ecg\n" + "\n".join(map(str, waveform())), encoding="utf-8")
    signal = CsvECGInputAdapter().load(file_path=path, sampling_rate_hz=100.0)
    assert signal.sample_count == 500
    assert signal.source_format == "csv"


def test_multi_column_csv_accepts_explicit_signal_column(tmp_path: Path) -> None:
    path = tmp_path / "record.csv"
    rows = [f"{index},{value}" for index, value in enumerate(waveform())]
    path.write_text("time,ecg\n" + "\n".join(rows), encoding="utf-8")
    signal = CsvECGInputAdapter().load(
        file_path=path,
        sampling_rate_hz=100.0,
        signal_column="ecg",
    )
    assert signal.sample_count == 500


def test_multi_column_csv_requires_signal_column(tmp_path: Path) -> None:
    path = tmp_path / "record.csv"
    path.write_text("a,b\n" + "\n".join(f"{x},{x + 1}" for x in range(500)))
    with pytest.raises(ValueError, match="multiple numeric columns"):
        CsvECGInputAdapter().load(file_path=path, sampling_rate_hz=100.0)


def test_valid_text_signal(tmp_path: Path) -> None:
    path = tmp_path / "record.txt"
    np.savetxt(path, waveform())
    signal = TextECGInputAdapter().load(file_path=path, sampling_rate_hz=100.0)
    assert signal.sample_count == 500
    assert signal.source_format == "txt"


def write_wfdb(tmp_path: Path, leads: tuple[str, ...] = ("I", "II", "MLII")) -> Path:
    values = np.column_stack([waveform() + offset for offset in range(len(leads))])
    wfdb.wrsamp(
        "record",
        fs=100,
        units=["mV"] * len(leads),
        sig_name=list(leads),
        p_signal=values,
        write_dir=str(tmp_path),
    )
    return tmp_path / "record.hea"


def test_wfdb_pair_loading_and_lead_priority(tmp_path: Path) -> None:
    header = write_wfdb(tmp_path)
    signal = WfdbECGInputAdapter().load(
        file_path=header,
        companion_file_path=header.with_suffix(".dat"),
        source_dataset=SourceDataset.MIT_BIH_ARRHYTHMIA,
    )
    assert signal.lead_name == "II"
    assert signal.lead_names == ("I", "II", "MLII")
    assert signal.units == "mV"
    assert signal.original_sampling_rate_hz == 100.0
    assert signal.source_dataset is SourceDataset.MIT_BIH_ARRHYTHMIA


def test_wfdb_explicit_lead_selection(tmp_path: Path) -> None:
    header = write_wfdb(tmp_path)
    signal = WfdbECGInputAdapter().load(
        file_path=header,
        companion_file_path=header.with_suffix(".dat"),
        lead_name="MLII",
    )
    assert signal.lead_name == "MLII"


@pytest.mark.parametrize(
    "source_dataset",
    [SourceDataset.PTB_XL, SourceDataset.PRIVATE, SourceDataset.UNKNOWN, None],
)
def test_wfdb_preserves_external_or_missing_provenance(
    tmp_path: Path,
    source_dataset: SourceDataset | None,
) -> None:
    header = write_wfdb(tmp_path)
    signal = WfdbECGInputAdapter().load(
        file_path=header,
        companion_file_path=header.with_suffix(".dat"),
        source_dataset=source_dataset,
    )
    assert signal.source_dataset is source_dataset


def test_wfdb_falls_back_to_first_lead_with_warning(tmp_path: Path) -> None:
    header = write_wfdb(tmp_path, ("V1", "V2"))
    signal = WfdbECGInputAdapter().load(
        file_path=header,
        companion_file_path=header.with_suffix(".dat"),
    )
    assert signal.lead_name == "V1"
    assert signal.warnings == (
        "No Lead II or MLII found; selected first lead 'V1'.",
    )


def test_wfdb_invalid_lead_name(tmp_path: Path) -> None:
    header = write_wfdb(tmp_path)
    with pytest.raises(ValueError, match="Available leads"):
        WfdbECGInputAdapter().load(
            file_path=header,
            companion_file_path=header.with_suffix(".dat"),
            lead_name="V9",
        )


@pytest.mark.parametrize(
    ("primary", "companion", "message"),
    [("record.dat", None, "header"), ("record.hea", None, "data")],
)
def test_wfdb_requires_both_files(
    primary: str, companion: str | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        WfdbECGInputAdapter().load(
            file_path=primary,
            companion_file_path=companion,
        )


@pytest.mark.parametrize(
    ("adapter", "suffix"),
    [(CsvECGInputAdapter(), "csv"), (TextECGInputAdapter(), "txt")],
)
def test_missing_sampling_rate(adapter: object, suffix: str, tmp_path: Path) -> None:
    path = tmp_path / f"record.{suffix}"
    if suffix == "csv":
        path.write_text("ecg\n" + "\n".join(map(str, waveform())))
    else:
        np.savetxt(path, waveform())
    with pytest.raises(ValueError, match="sampling_rate_hz is required"):
        adapter.load(file_path=path)  # type: ignore[attr-defined]


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_input_is_rejected(tmp_path: Path, bad_value: float) -> None:
    values = waveform()
    values[20] = bad_value
    path = tmp_path / "record.npy"
    np.save(path, values)
    with pytest.raises(InvalidSignalError, match="NaN or infinite"):
        NpyECGInputAdapter().load(file_path=path, sampling_rate_hz=100.0)


def test_empty_signal_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "record.npy"
    np.save(path, np.array([]))
    with pytest.raises(InvalidSignalError, match="cannot be empty"):
        NpyECGInputAdapter().load(file_path=path, sampling_rate_hz=100.0)


def test_unsupported_dimensionality_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "record.npy"
    np.save(path, np.ones((100, 2)))
    with pytest.raises(InvalidSignalError, match="one-dimensional"):
        NpyECGInputAdapter().load(file_path=path, sampling_rate_hz=100.0)
