from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, find_peaks, sosfiltfilt

from reasoning_pipeline.domain.enums.statuses import SignalQualityStatus
from reasoning_pipeline.domain.exceptions.pipeline_errors import (
    FeatureExtractionError,
    InsufficientSignalQualityError,
    UnsupportedSamplingRateError,
)
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.domain.models.r_peak_series import RPeakSeries
from reasoning_pipeline.domain.models.signal_quality import SignalQuality

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class _PeakCandidates:
    indices: IntArray
    prominences: FloatArray

    @property
    def count(self) -> int:
        return int(self.indices.size)


class SciPyRPeakDetector:
    """
    Detect R peaks using a deterministic SciPy-based QRS pipeline.

    Processing stages:

    1. Band-pass filtering.
    2. Derivative-based QRS enhancement.
    3. Squaring and moving-window integration.
    4. Adaptive QRS-region detection.
    5. Local R-peak refinement on the filtered ECG.
    6. RR interval and confidence calculation.

    The detector is replaceable and returns the standard RPeakSeries
    domain model used by downstream reasoning components.
    """

    DETECTOR_NAME = "scipy_adaptive_r_peak"
    DETECTOR_VERSION = "1.1.0"

    def __init__(
        self,
        *,
        lowcut_hz: float = 5.0,
        highcut_hz: float = 20.0,
        filter_order: int = 3,
        minimum_heart_rate_bpm: float = 30.0,
        maximum_heart_rate_bpm: float = 220.0,
        prominence_threshold: float = 0.60,
        minimum_required_peaks: int = 2,
        integration_window_seconds: float = 0.12,
        refinement_window_seconds: float = 0.12,
    ) -> None:
        self._validate_configuration(
            lowcut_hz=lowcut_hz,
            highcut_hz=highcut_hz,
            filter_order=filter_order,
            minimum_heart_rate_bpm=minimum_heart_rate_bpm,
            maximum_heart_rate_bpm=maximum_heart_rate_bpm,
            prominence_threshold=prominence_threshold,
            minimum_required_peaks=minimum_required_peaks,
            integration_window_seconds=integration_window_seconds,
            refinement_window_seconds=refinement_window_seconds,
        )

        self.lowcut_hz = lowcut_hz
        self.highcut_hz = highcut_hz
        self.filter_order = filter_order
        self.minimum_heart_rate_bpm = minimum_heart_rate_bpm
        self.maximum_heart_rate_bpm = maximum_heart_rate_bpm
        self.prominence_threshold = prominence_threshold
        self.minimum_required_peaks = minimum_required_peaks
        self.integration_window_seconds = integration_window_seconds
        self.refinement_window_seconds = refinement_window_seconds

    def detect(
        self,
        signal: ECGSignal,
        *,
        signal_quality: SignalQuality | None = None,
    ) -> RPeakSeries:
        """
        Detect R peaks in a validated single-lead ECG signal.
        """
        if (
            signal_quality is not None
            and signal_quality.status is SignalQualityStatus.UNUSABLE
        ):
            raise InsufficientSignalQualityError(
                "R-peak detection cannot run on an unusable ECG signal."
            )

        samples = np.asarray(signal.samples, dtype=np.float64)

        self._validate_signal(
            samples=samples,
            sampling_rate_hz=signal.sampling_rate_hz,
        )

        filtered_signal = self._bandpass_filter(
            samples=samples,
            sampling_rate_hz=signal.sampling_rate_hz,
        )

        self._validate_filtered_energy(
            original_signal=samples,
            filtered_signal=filtered_signal,
        )

        qrs_envelope = self._build_qrs_envelope(
            filtered_signal=filtered_signal,
            sampling_rate_hz=signal.sampling_rate_hz,
        )

        qrs_candidates = self._detect_qrs_candidates(
            qrs_envelope=qrs_envelope,
            sampling_rate_hz=signal.sampling_rate_hz,
        )

        refined_indices = self._refine_r_peaks(
            candidate_indices=qrs_candidates.indices,
            filtered_signal=filtered_signal,
            sampling_rate_hz=signal.sampling_rate_hz,
        )

        refined_indices = self._remove_refractory_duplicates(
            indices=refined_indices,
            filtered_signal=filtered_signal,
            sampling_rate_hz=signal.sampling_rate_hz,
        )

        if refined_indices.size < self.minimum_required_peaks:
            raise FeatureExtractionError(
                "Insufficient R peaks were detected for RR analysis."
            )

        timestamps_seconds = (
            refined_indices.astype(np.float64)
            / signal.sampling_rate_hz
        )

        rr_intervals_ms = (
            np.diff(refined_indices).astype(np.float64)
            / signal.sampling_rate_hz
            * 1000.0
        )

        confidence = self._calculate_confidence(
            indices=refined_indices,
            prominences=qrs_candidates.prominences,
            sampling_rate_hz=signal.sampling_rate_hz,
            duration_seconds=signal.duration_seconds,
            signal_quality=signal_quality,
        )

        return RPeakSeries(
            sample_indices=tuple(
                int(index) for index in refined_indices
            ),
            timestamps_seconds=tuple(
                float(timestamp) for timestamp in timestamps_seconds
            ),
            rr_intervals_ms=tuple(
                float(interval) for interval in rr_intervals_ms
            ),
            detector_name=self.DETECTOR_NAME,
            detector_version=self.DETECTOR_VERSION,
            confidence=confidence,
        )

    def _bandpass_filter(
        self,
        *,
        samples: FloatArray,
        sampling_rate_hz: float,
    ) -> FloatArray:
        nyquist_hz = sampling_rate_hz / 2.0

        if self.highcut_hz >= nyquist_hz:
            raise UnsupportedSamplingRateError(
                "Sampling rate is too low for the configured R-peak "
                "band-pass filter."
            )

        sections = butter(
            self.filter_order,
            [self.lowcut_hz, self.highcut_hz],
            btype="bandpass",
            fs=sampling_rate_hz,
            output="sos",
        )

        try:
            filtered = sosfiltfilt(sections, samples)
        except ValueError as error:
            raise FeatureExtractionError(
                "ECG signal is too short for zero-phase filtering."
            ) from error

        return np.asarray(filtered, dtype=np.float64)

    def _build_qrs_envelope(
        self,
        *,
        filtered_signal: FloatArray,
        sampling_rate_hz: float,
    ) -> FloatArray:
        derivative = np.diff(
            filtered_signal,
            prepend=filtered_signal[0],
        )

        squared_derivative = np.square(derivative)

        window_size = max(
            1,
            int(
                round(
                    self.integration_window_seconds
                    * sampling_rate_hz
                )
            ),
        )

        integration_kernel = (
            np.ones(window_size, dtype=np.float64)
            / window_size
        )

        envelope = np.convolve(
            squared_derivative,
            integration_kernel,
            mode="same",
        )

        return np.asarray(envelope, dtype=np.float64)

    def _detect_qrs_candidates(
        self,
        *,
        qrs_envelope: FloatArray,
        sampling_rate_hz: float,
    ) -> _PeakCandidates:
        envelope_median = float(np.median(qrs_envelope))

        envelope_mad = float(
            np.median(
                np.abs(qrs_envelope - envelope_median)
            )
        )

        robust_scale = 1.4826 * envelope_mad

        envelope_range = float(
            np.max(qrs_envelope) - envelope_median
        )

        adaptive_prominence = max(
            self.prominence_threshold * robust_scale,
            0.10 * envelope_range,
            np.finfo(np.float64).eps,
        )

        minimum_height = (
            envelope_median + 2.0 * robust_scale
        )

        minimum_distance_samples = max(
            1,
            int(
                round(
                    sampling_rate_hz
                    * 60.0
                    / self.maximum_heart_rate_bpm
                )
            ),
        )

        indices, properties = find_peaks(
            qrs_envelope,
            distance=minimum_distance_samples,
            prominence=adaptive_prominence,
            height=minimum_height,
        )

        prominences = np.asarray(
            properties["prominences"],
            dtype=np.float64,
        )

        return _PeakCandidates(
            indices=np.asarray(indices, dtype=np.int64),
            prominences=prominences,
        )

    def _refine_r_peaks(
        self,
        *,
        candidate_indices: IntArray,
        filtered_signal: FloatArray,
        sampling_rate_hz: float,
    ) -> IntArray:
        refinement_radius = max(
            1,
            int(
                round(
                    self.refinement_window_seconds
                    * sampling_rate_hz
                )
            ),
        )

        refined: list[int] = []

        for candidate_index in candidate_indices:
            start = max(
                0,
                int(candidate_index) - refinement_radius,
            )
            stop = min(
                filtered_signal.size,
                int(candidate_index) + refinement_radius + 1,
            )

            local_signal = filtered_signal[start:stop]

            if local_signal.size == 0:
                continue

            local_index = int(
                np.argmax(np.abs(local_signal))
            )

            refined.append(start + local_index)

        return np.asarray(refined, dtype=np.int64)

    def _remove_refractory_duplicates(
        self,
        *,
        indices: IntArray,
        filtered_signal: FloatArray,
        sampling_rate_hz: float,
    ) -> IntArray:
        if indices.size == 0:
            return indices

        minimum_distance_samples = max(
            1,
            int(
                round(
                    sampling_rate_hz
                    * 60.0
                    / self.maximum_heart_rate_bpm
                )
            ),
        )

        sorted_indices = sorted(
            set(int(index) for index in indices)
        )

        retained: list[int] = []

        for index in sorted_indices:
            if not retained:
                retained.append(index)
                continue

            previous_index = retained[-1]

            if (
                index - previous_index
                >= minimum_distance_samples
            ):
                retained.append(index)
                continue

            current_amplitude = abs(
                float(filtered_signal[index])
            )
            previous_amplitude = abs(
                float(filtered_signal[previous_index])
            )

            if current_amplitude > previous_amplitude:
                retained[-1] = index

        return np.asarray(retained, dtype=np.int64)

    @staticmethod
    def _validate_filtered_energy(
        *,
        original_signal: FloatArray,
        filtered_signal: FloatArray,
    ) -> None:
        original_median = float(
            np.median(original_signal)
        )

        original_mad = float(
            np.median(
                np.abs(original_signal - original_median)
            )
        )

        original_scale = 1.4826 * original_mad

        if original_scale <= 1e-12:
            original_scale = float(
                np.std(original_signal)
            )

        filtered_rms = float(
            np.sqrt(np.mean(np.square(filtered_signal)))
        )

        if (
            original_scale <= 1e-12
            or filtered_rms / original_scale < 0.01
        ):
            raise FeatureExtractionError(
                "Insufficient R peaks were detected for RR analysis."
            )

    def _calculate_confidence(
        self,
        *,
        indices: IntArray,
        prominences: FloatArray,
        sampling_rate_hz: float,
        duration_seconds: float,
        signal_quality: SignalQuality | None,
    ) -> float:
        estimated_heart_rate = (
            indices.size / duration_seconds * 60.0
        )

        heart_rate_score = self._heart_rate_plausibility(
            estimated_heart_rate
        )

        if prominences.size == 0:
            prominence_score = 0.0
        else:
            median_prominence = float(
                np.median(prominences)
            )
            maximum_prominence = float(
                np.max(prominences)
            )

            if maximum_prominence <= 0:
                prominence_score = 0.0
            else:
                prominence_score = float(
                    np.clip(
                        median_prominence
                        / maximum_prominence,
                        0.0,
                        1.0,
                    )
                )

        rr_intervals = (
            np.diff(indices).astype(np.float64)
            / sampling_rate_hz
        )

        regularity_score = self._calculate_regularity_score(
            rr_intervals
        )

        quality_score = (
            signal_quality.score
            if signal_quality is not None
            else 0.75
        )

        confidence = (
            0.30 * heart_rate_score
            + 0.25 * prominence_score
            + 0.20 * regularity_score
            + 0.25 * quality_score
        )

        return float(np.clip(confidence, 0.0, 1.0))

    def _heart_rate_plausibility(
        self,
        heart_rate_bpm: float,
    ) -> float:
        if (
            heart_rate_bpm < self.minimum_heart_rate_bpm
            or heart_rate_bpm > self.maximum_heart_rate_bpm
        ):
            return 0.0

        preferred_minimum = 40.0
        preferred_maximum = 180.0

        if (
            preferred_minimum
            <= heart_rate_bpm
            <= preferred_maximum
        ):
            return 1.0

        if heart_rate_bpm < preferred_minimum:
            denominator = (
                preferred_minimum
                - self.minimum_heart_rate_bpm
            )

            return float(
                np.clip(
                    (
                        heart_rate_bpm
                        - self.minimum_heart_rate_bpm
                    )
                    / denominator,
                    0.0,
                    1.0,
                )
            )

        denominator = (
            self.maximum_heart_rate_bpm
            - preferred_maximum
        )

        return float(
            np.clip(
                (
                    self.maximum_heart_rate_bpm
                    - heart_rate_bpm
                )
                / denominator,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _calculate_regularity_score(
        rr_intervals_seconds: FloatArray,
    ) -> float:
        if rr_intervals_seconds.size < 2:
            return 0.50

        mean_rr = float(
            np.mean(rr_intervals_seconds)
        )

        if mean_rr <= 0:
            return 0.0

        coefficient_of_variation = float(
            np.std(rr_intervals_seconds)
            / mean_rr
        )

        return float(
            np.clip(
                1.0
                - coefficient_of_variation / 0.30,
                0.0,
                1.0,
            )
        )

    def _validate_signal(
        self,
        *,
        samples: FloatArray,
        sampling_rate_hz: float,
    ) -> None:
        if samples.ndim != 1:
            raise FeatureExtractionError(
                "R-peak detection requires a one-dimensional ECG."
            )

        if samples.size < 3:
            raise FeatureExtractionError(
                "ECG signal is too short for R-peak detection."
            )

        if not np.all(np.isfinite(samples)):
            raise FeatureExtractionError(
                "ECG signal contains non-finite samples."
            )

        signal_range = float(
            np.max(samples) - np.min(samples)
        )

        if signal_range <= 1e-12:
            raise FeatureExtractionError(
                "ECG signal has negligible amplitude variation."
            )

        if (
            sampling_rate_hz <= 0
            or not np.isfinite(sampling_rate_hz)
        ):
            raise UnsupportedSamplingRateError(
                "Sampling rate must be finite and greater than zero."
            )

        if self.highcut_hz >= sampling_rate_hz / 2.0:
            raise UnsupportedSamplingRateError(
                "Sampling rate is too low for the configured filter."
            )

    @staticmethod
    def _validate_configuration(
        *,
        lowcut_hz: float,
        highcut_hz: float,
        filter_order: int,
        minimum_heart_rate_bpm: float,
        maximum_heart_rate_bpm: float,
        prominence_threshold: float,
        minimum_required_peaks: int,
        integration_window_seconds: float,
        refinement_window_seconds: float,
    ) -> None:
        if lowcut_hz <= 0:
            raise ValueError(
                "lowcut_hz must be greater than zero"
            )

        if highcut_hz <= lowcut_hz:
            raise ValueError(
                "highcut_hz must be greater than lowcut_hz"
            )

        if filter_order < 1:
            raise ValueError(
                "filter_order must be at least one"
            )

        if minimum_heart_rate_bpm <= 0:
            raise ValueError(
                "minimum_heart_rate_bpm must be greater than zero"
            )

        if (
            maximum_heart_rate_bpm
            <= minimum_heart_rate_bpm
        ):
            raise ValueError(
                "maximum_heart_rate_bpm must be greater than "
                "minimum_heart_rate_bpm"
            )

        if prominence_threshold <= 0:
            raise ValueError(
                "prominence_threshold must be greater than zero"
            )

        if minimum_required_peaks < 2:
            raise ValueError(
                "minimum_required_peaks must be at least two"
            )

        if integration_window_seconds <= 0:
            raise ValueError(
                "integration_window_seconds must be greater than zero"
            )

        if refinement_window_seconds <= 0:
            raise ValueError(
                "refinement_window_seconds must be greater than zero"
            )