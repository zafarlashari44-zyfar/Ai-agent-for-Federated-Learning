from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import neurokit2 as nk
import numpy as np
from numpy.typing import NDArray

from reasoning_pipeline.domain.models.ecg_signal import ECGSignal

Float64Array = NDArray[np.float64]


class SignalCleanerProtocol(Protocol):
    """Interface for ECG cleaning used before CNN beat segmentation."""

    def clean(
        self,
        signal: ECGSignal,
    ) -> Float64Array:
        ...


@dataclass(frozen=True)
class FrozenBaselineSignalCleaner:
    """
    Reproduce the signal-cleaning stage used by frozen Scribe v1.

    The original federated-learning preprocessing called:

        neurokit2.ecg_process(
            ecg_signal,
            sampling_rate=sampling_rate,
            method="neurokit",
        )

    and extracted CNN beat windows from the resulting ``ECG_Clean``
    column.

    R-peaks are intentionally not taken from ``ecg_process`` here.
    Scribe v2 remains responsible for R-peak detection and corrected
    peak locations. This class only reproduces the frozen cleaning
    transformation.
    """

    method: str = "neurokit"
    cleaning_version: str = "scribe-v1-neurokit"

    def clean(
        self,
        signal: ECGSignal,
    ) -> Float64Array:
        raw_samples = np.asarray(
            signal.samples,
            dtype=np.float64,
        )

        if raw_samples.ndim != 1:
            raise ValueError(
                "ECG signal samples must be one-dimensional."
            )

        if raw_samples.size == 0:
            raise ValueError(
                "ECG signal cannot be empty."
            )

        if not np.all(np.isfinite(raw_samples)):
            raise ValueError(
                "ECG signal contains NaN or infinite values."
            )

        sampling_rate = int(round(signal.sampling_rate_hz))

        try:
            processed_signals, _ = nk.ecg_process(
                raw_samples,
                sampling_rate=sampling_rate,
                method=self.method,
            )
        except Exception as error:
            raise RuntimeError(
                "Frozen baseline ECG cleaning failed."
            ) from error

        if "ECG_Clean" not in processed_signals.columns:
            raise RuntimeError(
                "NeuroKit ECG processing did not return ECG_Clean."
            )

        cleaned_samples = np.asarray(
            processed_signals["ECG_Clean"].to_numpy(),
            dtype=np.float64,
        )

        if cleaned_samples.shape != raw_samples.shape:
            raise RuntimeError(
                "Cleaned ECG signal length does not match the raw "
                "ECG signal length."
            )

        if not np.all(np.isfinite(cleaned_samples)):
            raise RuntimeError(
                "Cleaned ECG signal contains non-finite values."
            )

        return cleaned_samples
