from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reasoning_pipeline.domain.exceptions.pipeline_errors import (
    InvalidSignalError,
)
from reasoning_pipeline.scribe_v2 import NpyECGLoader


def test_loader_reads_valid_npy_file(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "record_001.npy"
    expected = np.array(
        [0.1, 0.2, 0.3, 0.4],
        dtype=np.float32,
    )

    np.save(file_path, expected)

    loader = NpyECGLoader()
    result = loader.load(file_path)

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, expected)


def test_loader_preserves_array_shape(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "multi_dimensional.npy"
    expected = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        dtype=np.float64,
    )

    np.save(file_path, expected)

    result = NpyECGLoader().load(file_path)

    assert result.shape == (2, 3)


def test_loader_rejects_missing_file(
    tmp_path: Path,
) -> None:
    missing_file = tmp_path / "missing.npy"

    with pytest.raises(
        FileNotFoundError,
        match="ECG file does not exist",
    ):
        NpyECGLoader().load(missing_file)


def test_loader_rejects_directory_path(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "ecg_directory"
    directory.mkdir()

    with pytest.raises(
        InvalidSignalError,
        match="ECG path is not a file",
    ):
        NpyECGLoader().load(directory)


def test_loader_rejects_unsupported_extension(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "record.csv"
    file_path.write_text(
        "0.1,0.2,0.3",
        encoding="utf-8",
    )

    with pytest.raises(
        InvalidSignalError,
        match="Unsupported ECG file format",
    ):
        NpyECGLoader().load(file_path)


def test_loader_rejects_corrupted_npy_file(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "corrupted.npy"
    file_path.write_bytes(b"this is not a valid numpy file")

    with pytest.raises(
        InvalidSignalError,
        match="Unable to load ECG NumPy file",
    ):
        NpyECGLoader().load(file_path)


def test_loader_rejects_object_array(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "object_array.npy"

    unsafe_array = np.array(
        [{"value": 1}],
        dtype=object,
    )
    np.save(file_path, unsafe_array)

    with pytest.raises(
        InvalidSignalError,
        match="Unable to load ECG NumPy file",
    ):
        NpyECGLoader().load(file_path)