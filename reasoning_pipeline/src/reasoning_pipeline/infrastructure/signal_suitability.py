from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from reasoning_pipeline.domain.enums.statuses import (
    SignalSuitabilityStatus,
    SourceDataset,
)
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.domain.models.signal_suitability_assessment import (
    SignalSuitabilityAssessment,
)


@dataclass(frozen=True)
class SignalSuitabilityThresholds:
    minimum_duration_seconds: float = 2.0
    unusual_short_duration_seconds: float = 10.0
    minimum_sample_count: int = 720
    minimum_variance_mv2: float = 1e-8
    flatline_difference_mv: float = 1e-5
    flatline_warning_percentage: float = 20.0
    flatline_rejection_percentage: float = 80.0
    clipping_warning_percentage: float = 10.0
    clipping_rejection_percentage: float = 50.0
    noise_warning_score: float = 0.65
    noise_rejection_score: float = 0.98
    expected_max_absolute_amplitude_mv: float = 10.0
    invalid_max_absolute_amplitude_mv: float = 100.0
    minimum_heart_rate_bpm: float = 30.0
    maximum_heart_rate_bpm: float = 220.0
    peak_minimum_distance_seconds: float = 0.25
    peak_prominence_standard_deviations: float = 0.5
    peak_minimum_prominence_mv: float = 0.02


class HeuristicSignalSuitabilityAssessor:
    def __init__(
        self,
        thresholds: SignalSuitabilityThresholds | None = None,
    ) -> None:
        self.thresholds = thresholds or SignalSuitabilityThresholds()

    def assess(self, signal: ECGSignal) -> SignalSuitabilityAssessment:
        samples = np.asarray(signal.samples, dtype=np.float64)
        warnings = list(signal.harmonisation_warnings)
        reasons: list[str] = []
        finite_ratio = float(np.mean(np.isfinite(samples))) if samples.size else 0.0
        rate = float(signal.sampling_rate_hz)
        valid_rate = np.isfinite(rate) and rate > 0
        duration = samples.size / rate if valid_rate else 0.0

        if samples.size == 0:
            reasons.append("Signal is empty.")
        if finite_ratio < 1.0:
            reasons.append("Signal contains non-finite samples.")
        if not valid_rate:
            reasons.append("Sampling metadata is invalid.")
        if samples.size < self.thresholds.minimum_sample_count:
            reasons.append(
                "Signal contains fewer than "
                f"{self.thresholds.minimum_sample_count} harmonised samples."
            )
        if duration < self.thresholds.minimum_duration_seconds:
            reasons.append(
                "Signal duration is too short for reliable beat analysis."
            )

        finite = samples[np.isfinite(samples)]
        variance = float(np.var(finite)) if finite.size else 0.0
        if variance <= self.thresholds.minimum_variance_mv2:
            reasons.append("Signal variance is effectively zero.")

        flatline = self._flatline_percentage(finite)
        clipping = self._clipping_percentage(finite)
        noise = self._noise_score(finite)
        max_amplitude = float(np.max(np.abs(finite))) if finite.size else 0.0

        if flatline >= self.thresholds.flatline_rejection_percentage:
            reasons.append("Flatline percentage exceeds the rejection threshold.")
        elif flatline >= self.thresholds.flatline_warning_percentage:
            warnings.append("Signal contains substantial flatline regions.")
        if clipping >= self.thresholds.clipping_rejection_percentage:
            reasons.append("Extreme clipping exceeds the rejection threshold.")
        elif clipping >= self.thresholds.clipping_warning_percentage:
            warnings.append("Signal contains possible clipping.")
        if noise >= self.thresholds.noise_rejection_score:
            reasons.append("Excessive high-frequency noise was detected.")
        elif noise >= self.thresholds.noise_warning_score:
            warnings.append("Signal quality is reduced by high-frequency noise.")
        if max_amplitude > self.thresholds.invalid_max_absolute_amplitude_mv:
            reasons.append("Amplitude range is implausible after mV conversion.")
        elif max_amplitude > self.thresholds.expected_max_absolute_amplitude_mv:
            warnings.append("Amplitude is outside the expected ECG range.")

        peaks, heart_rate = self._detect_r_peaks(finite, rate)
        if peaks == 0:
            reasons.append("No technically detectable R peaks were found.")
        if heart_rate is not None and not (
            self.thresholds.minimum_heart_rate_bpm
            <= heart_rate
            <= self.thresholds.maximum_heart_rate_bpm
        ):
            reasons.append("Estimated heart rate is technically implausible.")

        if signal.lead_name is not None and signal.lead_names:
            if signal.lead_name not in signal.lead_names:
                reasons.append("Selected lead is unavailable in source metadata.")
        elif signal.lead_name is None:
            if signal.source_format == "npy":
                warnings.append("Legacy NPY lead metadata is unavailable.")
            else:
                reasons.append("Selected lead metadata is unavailable.")

        if signal.resampled:
            warnings.append("Signal was resampled to the model sampling rate.")
        if signal.source_dataset is not SourceDataset.MIT_BIH_ARRHYTHMIA:
            warnings.append("Source is outside the validated MIT-BIH dataset.")
        if signal.lead_name not in {"MLII", "II", "Lead II"}:
            warnings.append("Selected lead is not MLII or Lead II.")
        if (
            duration < self.thresholds.unusual_short_duration_seconds
            and duration >= self.thresholds.minimum_duration_seconds
        ):
            warnings.append("Recording duration is unusually short.")

        quality_score = self._quality_score(flatline, clipping, noise)
        unique_warnings = tuple(dict.fromkeys(warnings))
        unique_reasons = tuple(dict.fromkeys(reasons))
        if unique_reasons:
            status = SignalSuitabilityStatus.REJECTED
        elif unique_warnings:
            status = SignalSuitabilityStatus.ACCEPTED_WITH_WARNINGS
        else:
            status = SignalSuitabilityStatus.ACCEPTED
        return SignalSuitabilityAssessment(
            status=status,
            suitable_for_processing=not unique_reasons,
            quality_score=quality_score,
            duration_seconds=duration,
            sampling_rate_hz=rate,
            selected_lead=signal.lead_name,
            units=signal.units,
            detected_r_peak_count=peaks,
            estimated_heart_rate_bpm=heart_rate,
            finite_sample_ratio=finite_ratio,
            flatline_percentage=flatline,
            clipping_percentage=clipping,
            noise_score=noise,
            warnings=unique_warnings,
            rejection_reasons=unique_reasons,
        )

    def _flatline_percentage(self, samples: NDArray[np.float64]) -> float:
        if samples.size < 2:
            return 100.0
        return float(
            np.mean(
                np.abs(np.diff(samples))
                <= self.thresholds.flatline_difference_mv
            )
            * 100.0
        )

    @staticmethod
    def _clipping_percentage(samples: NDArray[np.float64]) -> float:
        if samples.size == 0:
            return 100.0
        minimum = float(np.min(samples))
        maximum = float(np.max(samples))
        if maximum - minimum <= 1e-12:
            return 100.0
        tolerance = max((maximum - minimum) * 1e-6, 1e-12)
        return float(
            (
                np.mean(np.isclose(samples, minimum, atol=tolerance))
                + np.mean(np.isclose(samples, maximum, atol=tolerance))
            )
            * 100.0
        )

    @staticmethod
    def _noise_score(samples: NDArray[np.float64]) -> float:
        if samples.size < 3:
            return 1.0
        robust_scale = 1.4826 * float(
            np.median(np.abs(samples - np.median(samples)))
        )
        if robust_scale <= 1e-12:
            return 1.0
        roughness = float(np.median(np.abs(np.diff(samples, n=2))))
        return float(np.clip(roughness / robust_scale / 1.5, 0.0, 1.0))

    def _detect_r_peaks(
        self,
        samples: NDArray[np.float64],
        rate: float,
    ) -> tuple[int, float | None]:
        if samples.size < 3 or not np.isfinite(rate) or rate <= 0:
            return 0, None
        centered = np.abs(samples - np.median(samples))
        prominence = max(
            float(np.std(samples))
            * self.thresholds.peak_prominence_standard_deviations,
            self.thresholds.peak_minimum_prominence_mv,
        )
        indices, _ = find_peaks(
            centered,
            distance=max(
                1,
                int(rate * self.thresholds.peak_minimum_distance_seconds),
            ),
            prominence=prominence,
        )
        if indices.size < 2:
            return int(indices.size), None
        duration_minutes = (indices[-1] - indices[0]) / rate / 60.0
        heart_rate = (indices.size - 1) / duration_minutes
        return int(indices.size), float(heart_rate)

    @staticmethod
    def _quality_score(flatline: float, clipping: float, noise: float) -> float:
        penalty = 0.4 * min(flatline / 100.0, 1.0)
        penalty += 0.25 * min(clipping / 100.0, 1.0)
        penalty += 0.35 * noise
        return float(np.clip(1.0 - penalty, 0.0, 1.0))
