from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from reasoning_pipeline.domain.exceptions.pipeline_errors import (
    InvalidSignalError,
)

FloatArray = NDArray[np.float64]


class NpyECGLoader:
    """Load a single ECG signal from a NumPy .npy file."""

    SUPPORTED_SUFFIX = ".npy"

    def load(self, file_path: str | Path) -> FloatArray:
        """
        Load an ECG signal from a .npy file.

        The returned array is converted to float64 but is not otherwise
        validated or reshaped. Signal validation remains the responsibility
        of the validator component.

        Args:
            file_path:
                Path to the NumPy file containing the ECG signal.

        Returns:
            A NumPy float64 array containing the loaded ECG samples.

        Raises:
            FileNotFoundError:
                If the requested file does not exist.

            InvalidSignalError:
                If the path is not a file, has an unsupported extension,
                cannot be loaded, or does not contain a NumPy array.
        """
        path = Path(file_path).expanduser()

        if not path.exists():
            raise FileNotFoundError(f"ECG file does not exist: {path}")

        if not path.is_file():
            raise InvalidSignalError(
                f"ECG path is not a file: {path}"
            )

        if path.suffix.lower() != self.SUPPORTED_SUFFIX:
            raise InvalidSignalError(
                "Unsupported ECG file format. "
                f"Expected '{self.SUPPORTED_SUFFIX}', "
                f"received '{path.suffix or '<no extension>'}'."
            )

        try:
            loaded_data = np.load(
                path,
                allow_pickle=False,
            )
        except (OSError, ValueError, TypeError) as exc:
            raise InvalidSignalError(
                f"Unable to load ECG NumPy file: {path}"
            ) from exc

        if not isinstance(loaded_data, np.ndarray):
            raise InvalidSignalError(
                f"NumPy file does not contain an array: {path}"
            )

        try:
            return np.asarray(
                loaded_data,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise InvalidSignalError(
                "ECG array cannot be converted to numeric float64 values."
            ) from exc