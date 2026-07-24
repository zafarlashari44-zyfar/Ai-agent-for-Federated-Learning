from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt

from reasoning_pipeline.domain.exceptions.pipeline_errors import (
    FeatureExtractionError,
    UnsupportedSamplingRateError,
)
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.domain.models.morphology_features import (
    MorphologyFeatures,
)
from reasoning_pipeline.domain.models.r_peak_series import RPeakSeries

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class _BeatMorphology:
    """Internal morphology measurements for one detected beat."""

    qrs_duration_ms: float
    r_amplitude: float


class MorphologyFeatureExtractor:
    """
    Extract conservative single-lead ECG morphology measurements.

    Version 1 estimates:

    - Mean QRS duration
    - Mean baseline-corrected R-wave amplitude
    - Number of beats with QRS duration outside configured limits
    - Morphology extraction confidence

    PR and QT intervals are intentionally left unavailable because
    reliable P-wave and T-wave delineation requires a separately
    validated delineation method.

    The output provides technical evidence and is not a diagnosis.
    """

    EXTRACTOR_NAME = "single_lead_qrs_morphology"
    EXTRACTOR_VERSION = "1.0.0"

    def __init__(
        self,
        *,
        lowcut_hz: float = 0.5,
        highcut_hz: float = 40.0,
        filter_order: int = 3,
        pre_r_window_ms: float = 120.0,
        post_r_window_ms: float = 160.0,
        baseline_window_ms: float = 80.0,
        boundary_threshold_ratio: float = 0.12,
        boundary_stability_ms: float = 12.0,
        minimum_qrs_duration_ms: float = 50.0,
        maximum_qrs_duration_ms: float = 180.0,
        minimum_required_beats: int = 2,
    ) -> None:
        self._validate_configuration(
            lowcut_hz=lowcut_hz,
            highcut_hz=highcut_hz,
            filter_order=filter_order,
            pre_r_window_ms=pre_r_window_ms,
            post_r_window_ms=post_r_window_ms,
            baseline_window_ms=baseline_window_ms,
            boundary_threshold_ratio=boundary_threshold_ratio,
            boundary_stability_ms=boundary_stability_ms,
            minimum_qrs_duration_ms=minimum_qrs_duration_ms,
            maximum_qrs_duration_ms=maximum_qrs_duration_ms,
            minimum_required_beats=minimum_required_beats,
        )

        self.lowcut_hz = lowcut_hz
        self.highcut_hz = highcut_hz
        self.filter_order = filter_order
        self.pre_r_window_ms = pre_r_window_ms
        self.post_r_window_ms = post_r_window_ms
        self.baseline_window_ms = baseline_window_ms
        self.boundary_threshold_ratio = boundary_threshold_ratio
        self.boundary_stability_ms = boundary_stability_ms
        self.minimum_qrs_duration_ms = minimum_qrs_duration_ms
        self.maximum_qrs_duration_ms = maximum_qrs_duration_ms
        self.minimum_required_beats = minimum_required_beats

    def extract(
        self,
        signal: ECGSignal,
        r_peaks: RPeakSeries,
    ) -> MorphologyFeatures:
        """
        Extract aggregate morphology features from detected beats.
        """
        samples = np.asarray(signal.samples, dtype=np.float64)
        peak_indices = np.asarray(
            r_peaks.sample_indices,
            dtype=np.int64,
        )

        self._validate_inputs(
            samples=samples,
            peak_indices=peak_indices,
            sampling_rate_hz=signal.sampling_rate_hz,
        )

        filtered_signal = self._filter_signal(
            samples=samples,
            sampling_rate_hz=signal.sampling_rate_hz,
        )

        beat_measurements: list[_BeatMorphology] = []

        for peak_index in peak_indices:
            measurement = self._measure_beat(
                filtered_signal=filtered_signal,
                peak_index=int(peak_index),
                sampling_rate_hz=signal.sampling_rate_hz,
            )

            if measurement is not None:
                beat_measurements.append(measurement)

        if len(beat_measurements) < self.minimum_required_beats:
            raise FeatureExtractionError(
                "Insufficient valid beats were available for morphology "
                "analysis."
            )

        qrs_durations = np.asarray(
            [
                measurement.qrs_duration_ms
                for measurement in beat_measurements
            ],
            dtype=np.float64,
        )

        r_amplitudes = np.asarray(
            [
                measurement.r_amplitude
                for measurement in beat_measurements
            ],
            dtype=np.float64,
        )

        abnormal_beat_count = int(
            np.sum(
                (qrs_durations < self.minimum_qrs_duration_ms)
                | (qrs_durations > self.maximum_qrs_duration_ms)
            )
        )

        confidence = self._calculate_confidence(
            requested_beat_count=peak_indices.size,
            valid_beat_count=len(beat_measurements),
            qrs_durations=qrs_durations,
            detector_confidence=r_peaks.confidence,
        )

        return MorphologyFeatures(
            mean_qrs_duration_ms=float(np.mean(qrs_durations)),
            mean_pr_interval_ms=None,
            mean_qt_interval_ms=None,
            mean_r_amplitude=float(np.mean(r_amplitudes)),
            abnormal_beat_count=abnormal_beat_count,
            morphology_confidence=confidence,
        )

    def _measure_beat(
        self,
        *,
        filtered_signal: FloatArray,
        peak_index: int,
        sampling_rate_hz: float,
    ) -> _BeatMorphology | None:
        pre_samples = self._milliseconds_to_samples(
            self.pre_r_window_ms,
            sampling_rate_hz,
        )
        post_samples = self._milliseconds_to_samples(
            self.post_r_window_ms,
            sampling_rate_hz,
        )

        start = peak_index - pre_samples
        stop = peak_index + post_samples + 1

        if start < 0 or stop > filtered_signal.size:
            return None

        beat = filtered_signal[start:stop]

        local_peak_index = pre_samples

        baseline = self._estimate_baseline(
            beat=beat,
            local_peak_index=local_peak_index,
            sampling_rate_hz=sampling_rate_hz,
        )

        centered_beat = beat - baseline
        r_amplitude = float(centered_beat[local_peak_index])

        absolute_peak_amplitude = abs(r_amplitude)

        if absolute_peak_amplitude <= 1e-12:
            return None

        threshold = max(
            absolute_peak_amplitude * self.boundary_threshold_ratio,
            np.finfo(np.float64).eps,
        )

        stability_samples = max(
            1,
            self._milliseconds_to_samples(
                self.boundary_stability_ms,
                sampling_rate_hz,
            ),
        )

        onset_index = self._find_onset(
            centered_beat=centered_beat,
            peak_index=local_peak_index,
            threshold=threshold,
            stability_samples=stability_samples,
        )

        offset_index = self._find_offset(
            centered_beat=centered_beat,
            peak_index=local_peak_index,
            threshold=threshold,
            stability_samples=stability_samples,
        )

        if onset_index is None or offset_index is None:
            return None

        if offset_index <= onset_index:
            return None

        qrs_duration_ms = (
            (offset_index - onset_index)
            / sampling_rate_hz
            * 1000.0
        )

        if not np.isfinite(qrs_duration_ms):
            return None

        return _BeatMorphology(
            qrs_duration_ms=float(qrs_duration_ms),
            r_amplitude=r_amplitude,
        )

    def _estimate_baseline(
        self,
        *,
        beat: FloatArray,
        local_peak_index: int,
        sampling_rate_hz: float,
    ) -> float:
        baseline_samples = self._milliseconds_to_samples(
            self.baseline_window_ms,
            sampling_rate_hz,
        )

        baseline_stop = max(
            1,
            local_peak_index
            - self._milliseconds_to_samples(
                40.0,
                sampling_rate_hz,
            ),
        )

        baseline_start = max(
            0,
            baseline_stop - baseline_samples,
        )

        baseline_segment = beat[baseline_start:baseline_stop]

        if baseline_segment.size == 0:
            return float(np.median(beat))

        return float(np.median(baseline_segment))

    @staticmethod
    def _find_onset(
        *,
        centered_beat: FloatArray,
        peak_index: int,
        threshold: float,
        stability_samples: int,
    ) -> int | None:
        absolute_signal = np.abs(centered_beat)

        for index in range(
            peak_index - 1,
            stability_samples - 1,
            -1,
        ):
            segment = absolute_signal[
                index - stability_samples + 1:index + 1
            ]

            if np.all(segment <= threshold):
                return index

        return None

    @staticmethod
    def _find_offset(
        *,
        centered_beat: FloatArray,
        peak_index: int,
        threshold: float,
        stability_samples: int,
    ) -> int | None:
        absolute_signal = np.abs(centered_beat)

        maximum_start = absolute_signal.size - stability_samples

        for index in range(
            peak_index + 1,
            maximum_start + 1,
        ):
            segment = absolute_signal[
                index:index + stability_samples
            ]

            if np.all(segment <= threshold):
                return index

        return None

    def _filter_signal(
        self,
        *,
        samples: FloatArray,
        sampling_rate_hz: float,
    ) -> FloatArray:
        nyquist_hz = sampling_rate_hz / 2.0

        if self.highcut_hz >= nyquist_hz:
            raise UnsupportedSamplingRateError(
                "Sampling rate is too low for the configured morphology "
                "filter."
            )

        sections = butter(
            self.filter_order,
            [self.lowcut_hz, self.highcut_hz],
            btype="bandpass",
            fs=sampling_rate_hz,
            output="sos",
        )

        try:
            filtered = sosfiltfilt(
                sections,
                samples,
            )
        except ValueError as error:
            raise FeatureExtractionError(
                "ECG signal is too short for morphology filtering."
            ) from error

        return np.asarray(filtered, dtype=np.float64)

    @staticmethod
    def _calculate_confidence(
        *,
        requested_beat_count: int,
        valid_beat_count: int,
        qrs_durations: FloatArray,
        detector_confidence: float,
    ) -> float:
        coverage_score = (
            valid_beat_count / requested_beat_count
            if requested_beat_count > 0
            else 0.0
        )

        mean_duration = float(np.mean(qrs_durations))

        if mean_duration <= 0 or qrs_durations.size < 2:
            consistency_score = 0.5
        else:
            coefficient_of_variation = float(
                np.std(qrs_durations)
                / mean_duration
            )

            consistency_score = float(
                np.clip(
                    1.0 - coefficient_of_variation / 0.30,
                    0.0,
                    1.0,
                )
            )

        confidence = (
            0.40 * coverage_score
            + 0.35 * consistency_score
            + 0.25 * detector_confidence
        )

        return float(np.clip(confidence, 0.0, 1.0))

    @staticmethod
    def _milliseconds_to_samples(
        milliseconds: float,
        sampling_rate_hz: float,
    ) -> int:
        return max(
            1,
            int(
                round(
                    milliseconds
                    / 1000.0
                    * sampling_rate_hz
                )
            ),
        )

    def _validate_inputs(
        self,
        *,
        samples: FloatArray,
        peak_indices: NDArray[np.int64],
        sampling_rate_hz: float,
    ) -> None:
        if samples.ndim != 1:
            raise FeatureExtractionError(
                "Morphology extraction requires a one-dimensional ECG."
            )

        if samples.size < 3:
            raise FeatureExtractionError(
                "ECG signal is too short for morphology extraction."
            )

        if not np.all(np.isfinite(samples)):
            raise FeatureExtractionError(
                "ECG signal contains non-finite samples."
            )

        if (
            not np.isfinite(sampling_rate_hz)
            or sampling_rate_hz <= 0
        ):
            raise UnsupportedSamplingRateError(
                "Sampling rate must be finite and greater than zero."
            )

        if peak_indices.ndim != 1:
            raise FeatureExtractionError(
                "R-peak indices must be one-dimensional."
            )

        if peak_indices.size < self.minimum_required_beats:
            raise FeatureExtractionError(
                "Insufficient R peaks were provided for morphology "
                "analysis."
            )

        if np.any(peak_indices < 0):
            raise FeatureExtractionError(
                "R-peak indices cannot be negative."
            )

        if np.any(peak_indices >= samples.size):
            raise FeatureExtractionError(
                "An R-peak index exceeds the ECG signal length."
            )

        if peak_indices.size > 1:
            if np.any(np.diff(peak_indices) <= 0):
                raise FeatureExtractionError(
                    "R-peak indices must be strictly increasing."
                )

    @staticmethod
    def _validate_configuration(
        *,
        lowcut_hz: float,
        highcut_hz: float,
        filter_order: int,
        pre_r_window_ms: float,
        post_r_window_ms: float,
        baseline_window_ms: float,
        boundary_threshold_ratio: float,
        boundary_stability_ms: float,
        minimum_qrs_duration_ms: float,
        maximum_qrs_duration_ms: float,
        minimum_required_beats: int,
    ) -> None:
        float_values = (
            lowcut_hz,
            highcut_hz,
            pre_r_window_ms,
            post_r_window_ms,
            baseline_window_ms,
            boundary_threshold_ratio,
            boundary_stability_ms,
            minimum_qrs_duration_ms,
            maximum_qrs_duration_ms,
        )

        if not all(np.isfinite(value) for value in float_values):
            raise ValueError(
                "Morphology extractor configuration must be finite."
            )

        if lowcut_hz <= 0:
            raise ValueError("lowcut_hz must be greater than zero")

        if highcut_hz <= lowcut_hz:
            raise ValueError(
                "highcut_hz must be greater than lowcut_hz"
            )

        if filter_order < 1:
            raise ValueError("filter_order must be at least one")

        if pre_r_window_ms <= 0:
            raise ValueError(
                "pre_r_window_ms must be greater than zero"
            )

        if post_r_window_ms <= 0:
            raise ValueError(
                "post_r_window_ms must be greater than zero"
            )

        if baseline_window_ms <= 0:
            raise ValueError(
                "baseline_window_ms must be greater than zero"
            )

        if not 0.0 < boundary_threshold_ratio < 1.0:
            raise ValueError(
                "boundary_threshold_ratio must be between zero and one"
            )

        if boundary_stability_ms <= 0:
            raise ValueError(
                "boundary_stability_ms must be greater than zero"
            )

        if minimum_qrs_duration_ms <= 0:
            raise ValueError(
                "minimum_qrs_duration_ms must be greater than zero"
            )

        if (
            maximum_qrs_duration_ms
            <= minimum_qrs_duration_ms
        ):
            raise ValueError(
                "maximum_qrs_duration_ms must be greater than "
                "minimum_qrs_duration_ms"
            )

        if minimum_required_beats < 2:
            raise ValueError(
                "minimum_required_beats must be at least two"
            )