from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.domain.models.feature_set import FeatureSet
from reasoning_pipeline.orchestration.baseline_signal_cleaner import (
    FrozenBaselineSignalCleaner,
    SignalCleanerProtocol,
)

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]


@dataclass(frozen=True)
class PreparedBeat:
    """One fixed-length CNN input extracted around an R-peak."""

    beat_index: int
    r_peak_sample_index: int
    source_start_sample_index: int
    source_stop_sample_index_exclusive: int
    r_peak_timestamp_seconds: float
    source_start_timestamp_seconds: float
    source_stop_timestamp_seconds_exclusive: float
    sampling_rate_hz: float
    samples: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be greater than zero")

        if (
            self.source_stop_sample_index_exclusive
            - self.source_start_sample_index
            != 216
        ):
            raise ValueError("Prepared beat source window must contain 216 samples")

        if (
            self.r_peak_sample_index - self.source_start_sample_index
            != 72
        ):
            raise ValueError(
                "Prepared beat R-peak must be 72 samples after window start"
            )

        expected_timestamps = (
            (
                self.r_peak_timestamp_seconds,
                self.r_peak_sample_index,
            ),
            (
                self.source_start_timestamp_seconds,
                self.source_start_sample_index,
            ),
            (
                self.source_stop_timestamp_seconds_exclusive,
                self.source_stop_sample_index_exclusive,
            ),
        )

        for timestamp, sample_index in expected_timestamps:
            if not np.isclose(
                timestamp,
                sample_index / self.sampling_rate_hz,
            ):
                raise ValueError(
                    "Prepared beat timestamps must equal sample index divided "
                    "by sampling_rate_hz"
                )

        if len(self.samples) != 216:
            raise ValueError("Prepared beat must contain exactly 216 samples")


class ModelInputPreparer:
    """
    Prepare CNN inputs using the frozen Scribe v1 preprocessing contract.

    Contract:

    - Input ECG sampling rate must be 360 Hz.
    - Clean the complete ECG using NeuroKit ``ecg_process`` with
      ``method="neurokit"``.
    - Extract beat windows from ``ECG_Clean``.
    - Use corrected R-peak indices produced by Scribe v2.
    - Extract 72 samples before each R-peak.
    - Extract 144 samples after each R-peak.
    - Produce exactly 216 samples per beat.
    - Apply independent per-beat z-score normalisation.
    - Skip R-peaks too close to the signal boundaries.
    """

    EXPECTED_SAMPLING_RATE_HZ = 360.0
    PRE_R_SAMPLES = 72
    POST_R_SAMPLES = 144
    INPUT_LENGTH = PRE_R_SAMPLES + POST_R_SAMPLES
    STANDARD_DEVIATION_EPSILON = 1e-8
    PREPROCESSING_VERSION = "scribe-v1-neurokit"

    def __init__(
        self,
        cleaner: SignalCleanerProtocol | None = None,
    ) -> None:
        self.cleaner = cleaner or FrozenBaselineSignalCleaner()

    def prepare_all(
        self,
        *,
        signal: ECGSignal,
        features: FeatureSet,
    ) -> tuple[PreparedBeat, ...]:
        """
        Clean the ECG, then extract and normalise every valid beat.

        Raises:
            ValueError:
                If the sampling rate is unsupported, no R-peaks are
                available, or no valid windows can be extracted.

            RuntimeError:
                If frozen baseline ECG cleaning fails.
        """
        self._validate_sampling_rate(signal.sampling_rate_hz)

        peak_indices = features.r_peaks.sample_indices

        if not peak_indices:
            raise ValueError(
                "No R-peaks are available for model input preparation."
            )

        cleaned_samples = self.cleaner.clean(signal)

        if cleaned_samples.shape != (len(signal.samples),):
            raise RuntimeError(
                "The cleaned ECG must have the same one-dimensional "
                "shape as the source ECG."
            )

        if not np.all(np.isfinite(cleaned_samples)):
            raise RuntimeError(
                "The cleaned ECG contains non-finite values."
            )

        prepared_beats: list[PreparedBeat] = []

        for beat_index, peak_index in enumerate(peak_indices):
            start = peak_index - self.PRE_R_SAMPLES
            stop = peak_index + self.POST_R_SAMPLES
            beat = self._extract_beat(
                samples=cleaned_samples,
                peak_index=peak_index,
            )

            if beat is None:
                continue

            normalised_beat = self._normalise_beat(beat)

            prepared_beats.append(
                PreparedBeat(
                    beat_index=beat_index,
                    r_peak_sample_index=peak_index,
                    source_start_sample_index=start,
                    source_stop_sample_index_exclusive=stop,
                    r_peak_timestamp_seconds=(
                        peak_index / signal.sampling_rate_hz
                    ),
                    source_start_timestamp_seconds=(
                        start / signal.sampling_rate_hz
                    ),
                    source_stop_timestamp_seconds_exclusive=(
                        stop / signal.sampling_rate_hz
                    ),
                    sampling_rate_hz=signal.sampling_rate_hz,
                    samples=tuple(
                        float(value)
                        for value in normalised_beat
                    ),
                )
            )

        if not prepared_beats:
            raise ValueError(
                "No valid 216-sample beats could be extracted. "
                "All detected R-peaks may be too close to the signal "
                "edges."
            )

        return tuple(prepared_beats)

    def prepare_representative(
        self,
        *,
        signal: ECGSignal,
        features: FeatureSet,
    ) -> PreparedBeat:
        """
        Return the temporal middle beat from all valid extracted beats.
        """
        prepared_beats = self.prepare_all(
            signal=signal,
            features=features,
        )

        middle_index = len(prepared_beats) // 2
        return prepared_beats[middle_index]

    def prepare_matrix(
        self,
        *,
        signal: ECGSignal,
        features: FeatureSet,
    ) -> Float32Array:
        """Return all valid beats as a two-dimensional float32 array."""
        prepared_beats = self.prepare_all(
            signal=signal,
            features=features,
        )

        matrix = np.asarray(
            [beat.samples for beat in prepared_beats],
            dtype=np.float32,
        )

        expected_shape = (
            len(prepared_beats),
            self.INPUT_LENGTH,
        )

        if matrix.shape != expected_shape:
            raise RuntimeError(
                "Prepared beat matrix has an unexpected shape. "
                f"Expected {expected_shape}, received {matrix.shape}."
            )

        return matrix

    def _extract_beat(
        self,
        *,
        samples: Float64Array,
        peak_index: int,
    ) -> Float64Array | None:
        start = peak_index - self.PRE_R_SAMPLES
        stop = peak_index + self.POST_R_SAMPLES

        if start < 0 or stop > samples.shape[0]:
            return None

        beat = np.asarray(
            samples[start:stop],
            dtype=np.float64,
        )

        if beat.shape != (self.INPUT_LENGTH,):
            return None

        if not np.all(np.isfinite(beat)):
            raise ValueError(
                "Extracted ECG beat contains non-finite values."
            )

        return beat

    def _normalise_beat(
        self,
        beat: Float64Array,
    ) -> Float32Array:
        mean = float(np.mean(beat))
        standard_deviation = float(np.std(beat))

        centred = beat - mean

        if standard_deviation > self.STANDARD_DEVIATION_EPSILON:
            normalised = centred / standard_deviation
        else:
            normalised = centred

        return np.asarray(
            normalised,
            dtype=np.float32,
        )

    def _validate_sampling_rate(
        self,
        sampling_rate_hz: float,
    ) -> None:
        if not np.isclose(
            sampling_rate_hz,
            self.EXPECTED_SAMPLING_RATE_HZ,
        ):
            raise ValueError(
                "The frozen federated CNN currently supports only "
                f"{self.EXPECTED_SAMPLING_RATE_HZ:g} Hz ECG signals. "
                f"Received {sampling_rate_hz:g} Hz."
            )
