from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reasoning_pipeline.domain.enums.statuses import SignalQualityStatus
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.domain.models.signal_quality import SignalQuality

FloatArray = NDArray[np.float64]


class ECGSignalQualityAssessor:
    """
    Estimate the technical quality of a single-lead ECG signal.

    The assessor measures several signal-quality indicators:

    - valid sample ratio
    - flatline proportion
    - repeated extreme-value proportion
    - high-frequency noise
    - baseline instability

    The output is intended to support downstream reasoning and
    evidence generation. It must not be interpreted as a clinical
    diagnosis or medical-device quality certification.
    """

    def __init__(
        self,
        *,
        good_threshold: float = 0.80,
        acceptable_threshold: float = 0.60,
        poor_threshold: float = 0.35,
        minimum_valid_sample_ratio: float = 0.80,
        maximum_flatline_ratio: float = 0.80,
    ) -> None:
        self._validate_thresholds(
            good_threshold=good_threshold,
            acceptable_threshold=acceptable_threshold,
            poor_threshold=poor_threshold,
            minimum_valid_sample_ratio=minimum_valid_sample_ratio,
            maximum_flatline_ratio=maximum_flatline_ratio,
        )

        self.good_threshold = good_threshold
        self.acceptable_threshold = acceptable_threshold
        self.poor_threshold = poor_threshold
        self.minimum_valid_sample_ratio = minimum_valid_sample_ratio
        self.maximum_flatline_ratio = maximum_flatline_ratio

    def assess(self, signal: ECGSignal) -> SignalQuality:
        """
        Assess the technical quality of an ECG signal.

        Args:
            signal:
                Standard ECGSignal domain object.

        Returns:
            SignalQuality containing a normalized score, quality status,
            estimated noise score, valid sample ratio, and warnings.
        """
        samples = np.asarray(signal.samples, dtype=np.float64)

        valid_mask = np.isfinite(samples)
        valid_sample_ratio = float(np.mean(valid_mask))

        if not np.any(valid_mask):
            return SignalQuality(
                score=0.0,
                status=SignalQualityStatus.UNUSABLE,
                noise_score=1.0,
                valid_sample_ratio=0.0,
                warnings=("Signal contains no finite samples.",),
            )

        valid_samples = samples[valid_mask]

        robust_scale = self._calculate_robust_scale(valid_samples)
        flatline_ratio = self._calculate_flatline_ratio(
            valid_samples,
            robust_scale,
        )
        clipping_ratio = self._calculate_clipping_ratio(valid_samples)
        noise_score = self._calculate_noise_score(
            valid_samples,
            robust_scale,
        )
        baseline_instability_score = (
            self._calculate_baseline_instability_score(
                valid_samples,
                signal.sampling_rate_hz,
                robust_scale,
            )
        )

        quality_score = self._calculate_quality_score(
            valid_sample_ratio=valid_sample_ratio,
            flatline_ratio=flatline_ratio,
            clipping_ratio=clipping_ratio,
            noise_score=noise_score,
            baseline_instability_score=baseline_instability_score,
        )

        warnings = self._build_warnings(
            valid_sample_ratio=valid_sample_ratio,
            flatline_ratio=flatline_ratio,
            clipping_ratio=clipping_ratio,
            noise_score=noise_score,
            baseline_instability_score=baseline_instability_score,
            robust_scale=robust_scale,
        )

        status = self._determine_status(
            quality_score=quality_score,
            valid_sample_ratio=valid_sample_ratio,
            flatline_ratio=flatline_ratio,
            robust_scale=robust_scale,
        )

        return SignalQuality(
            score=quality_score,
            status=status,
            noise_score=noise_score,
            valid_sample_ratio=valid_sample_ratio,
            warnings=warnings,
        )

    @staticmethod
    def _calculate_robust_scale(samples: FloatArray) -> float:
        median = float(np.median(samples))
        median_absolute_deviation = float(
            np.median(np.abs(samples - median))
        )

        robust_scale = 1.4826 * median_absolute_deviation

        if robust_scale <= 1e-12:
            robust_scale = float(np.std(samples))

        return max(robust_scale, 0.0)

    @staticmethod
    def _calculate_flatline_ratio(
        samples: FloatArray,
        robust_scale: float,
    ) -> float:
        if samples.size < 2:
            return 1.0

        tolerance = max(robust_scale * 1e-3, 1e-12)
        differences = np.abs(np.diff(samples))

        return float(np.mean(differences <= tolerance))

    @staticmethod
    def _calculate_clipping_ratio(samples: FloatArray) -> float:
        if samples.size == 0:
            return 1.0

        minimum = float(np.min(samples))
        maximum = float(np.max(samples))
        signal_range = maximum - minimum

        if signal_range <= 1e-12:
            return 1.0

        tolerance = max(signal_range * 1e-6, 1e-12)

        minimum_ratio = float(
            np.mean(np.isclose(samples, minimum, atol=tolerance, rtol=0.0))
        )
        maximum_ratio = float(
            np.mean(np.isclose(samples, maximum, atol=tolerance, rtol=0.0))
        )

        return min(1.0, minimum_ratio + maximum_ratio)

    @staticmethod
    def _calculate_noise_score(
        samples: FloatArray,
        robust_scale: float,
    ) -> float:
        if samples.size < 3 or robust_scale <= 1e-12:
            return 1.0

        first_difference = np.diff(samples)
        second_difference = np.diff(first_difference)

        roughness = float(np.median(np.abs(second_difference)))
        normalized_roughness = roughness / robust_scale

        return float(np.clip(normalized_roughness / 0.75, 0.0, 1.0))

    @staticmethod
    def _calculate_baseline_instability_score(
        samples: FloatArray,
        sampling_rate_hz: float,
        robust_scale: float,
    ) -> float:
        if samples.size < 3 or robust_scale <= 1e-12:
            return 1.0

        window_size = max(3, int(round(sampling_rate_hz)))

        if window_size >= samples.size:
            window_size = max(3, samples.size // 3)

        if window_size >= samples.size or window_size < 3:
            return 0.0

        kernel = np.ones(window_size, dtype=np.float64) / window_size

        baseline = np.convolve(
            samples,
            kernel,
            mode="valid",
        )

        baseline_variation = float(np.std(baseline))
        normalized_variation = baseline_variation / robust_scale

        return float(np.clip(normalized_variation / 0.50, 0.0, 1.0))

    @staticmethod
    def _calculate_quality_score(
        *,
        valid_sample_ratio: float,
        flatline_ratio: float,
        clipping_ratio: float,
        noise_score: float,
        baseline_instability_score: float,
    ) -> float:
        flatline_penalty = min(1.0, flatline_ratio / 0.25)
        clipping_penalty = min(1.0, clipping_ratio / 0.10)

        penalty = (
            0.35 * flatline_penalty
            + 0.20 * clipping_penalty
            + 0.30 * noise_score
            + 0.15 * baseline_instability_score
        )

        score = valid_sample_ratio * (1.0 - penalty)

        return float(np.clip(score, 0.0, 1.0))

    def _determine_status(
        self,
        *,
        quality_score: float,
        valid_sample_ratio: float,
        flatline_ratio: float,
        robust_scale: float,
    ) -> SignalQualityStatus:
        if (
            valid_sample_ratio < self.minimum_valid_sample_ratio
            or flatline_ratio >= self.maximum_flatline_ratio
            or robust_scale <= 1e-12
        ):
            return SignalQualityStatus.UNUSABLE

        if flatline_ratio >= 0.50:
            return SignalQualityStatus.POOR

        if quality_score >= self.good_threshold:
            return SignalQualityStatus.GOOD

        if quality_score >= self.acceptable_threshold:
            return SignalQualityStatus.ACCEPTABLE

        if quality_score >= self.poor_threshold:
            return SignalQualityStatus.POOR

        return SignalQualityStatus.UNUSABLE

    def _build_warnings(
        self,
        *,
        valid_sample_ratio: float,
        flatline_ratio: float,
        clipping_ratio: float,
        noise_score: float,
        baseline_instability_score: float,
        robust_scale: float,
    ) -> tuple[str, ...]:
        warnings: list[str] = []

        if valid_sample_ratio < 1.0:
            warnings.append(
                "Signal contains missing or non-finite samples."
            )

        if valid_sample_ratio < self.minimum_valid_sample_ratio:
            warnings.append(
                "Too many invalid samples are present for reliable analysis."
            )

        if robust_scale <= 1e-12:
            warnings.append(
                "Signal has negligible amplitude variation."
            )

        if flatline_ratio >= 0.20:
            warnings.append(
                "A substantial flatline segment may be present."
            )

        if flatline_ratio >= self.maximum_flatline_ratio:
            warnings.append(
                "Signal is predominantly flat and is considered unusable."
            )

        if clipping_ratio >= 0.05:
            warnings.append(
                "Repeated minimum or maximum values suggest possible clipping."
            )

        if noise_score >= 0.70:
            warnings.append(
                "High-frequency noise may reduce feature reliability."
            )

        if baseline_instability_score >= 0.70:
            warnings.append(
                "Baseline instability or drift may be present."
            )

        return tuple(warnings)

    @staticmethod
    def _validate_thresholds(
        *,
        good_threshold: float,
        acceptable_threshold: float,
        poor_threshold: float,
        minimum_valid_sample_ratio: float,
        maximum_flatline_ratio: float,
    ) -> None:
        values = {
            "good_threshold": good_threshold,
            "acceptable_threshold": acceptable_threshold,
            "poor_threshold": poor_threshold,
            "minimum_valid_sample_ratio": minimum_valid_sample_ratio,
            "maximum_flatline_ratio": maximum_flatline_ratio,
        }

        for name, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between zero and one"
                )

        if not (
            good_threshold
            > acceptable_threshold
            > poor_threshold
        ):
            raise ValueError(
                "Quality thresholds must satisfy "
                "good > acceptable > poor"
            )