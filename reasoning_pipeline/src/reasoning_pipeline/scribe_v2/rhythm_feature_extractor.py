from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from reasoning_pipeline.domain.exceptions.pipeline_errors import (
    FeatureExtractionError,
)
from reasoning_pipeline.domain.models.r_peak_series import RPeakSeries
from reasoning_pipeline.domain.models.rhythm_features import RhythmFeatures

FloatArray = NDArray[np.float64]


class RhythmFeatureExtractor:
    """
    Extract interpretable rhythm and heart-rate variability features.

    The extractor consumes an RPeakSeries and calculates:

    - Mean, minimum, and maximum instantaneous heart rate
    - Mean RR interval
    - SDNN
    - RMSSD
    - pNN50
    - A normalized rhythm irregularity score

    The irregularity score is a technical evidence feature. It must not
    be interpreted independently as a clinical diagnosis.
    """

    EXTRACTOR_NAME = "rr_rhythm_feature_extractor"
    EXTRACTOR_VERSION = "1.0.0"

    def __init__(
        self,
        *,
        minimum_rr_ms: float = 250.0,
        maximum_rr_ms: float = 2500.0,
        cv_reference: float = 0.20,
        rmssd_reference_ratio: float = 0.15,
        pnn50_reference_percent: float = 50.0,
    ) -> None:
        self._validate_configuration(
            minimum_rr_ms=minimum_rr_ms,
            maximum_rr_ms=maximum_rr_ms,
            cv_reference=cv_reference,
            rmssd_reference_ratio=rmssd_reference_ratio,
            pnn50_reference_percent=pnn50_reference_percent,
        )

        self.minimum_rr_ms = minimum_rr_ms
        self.maximum_rr_ms = maximum_rr_ms
        self.cv_reference = cv_reference
        self.rmssd_reference_ratio = rmssd_reference_ratio
        self.pnn50_reference_percent = pnn50_reference_percent

    def extract(
        self,
        r_peaks: RPeakSeries,
    ) -> RhythmFeatures:
        """
        Calculate rhythm features from detected R peaks.

        A single RR interval is sufficient for heart-rate and mean-RR
        calculations. At least two RR intervals are required for SDNN,
        RMSSD, pNN50, and the irregularity score.
        """
        rr_intervals = np.asarray(
            r_peaks.rr_intervals_ms,
            dtype=np.float64,
        )

        self._validate_r_peak_series(
            r_peaks=r_peaks,
            rr_intervals=rr_intervals,
        )

        if rr_intervals.size == 0:
            return self._empty_features()

        heart_rates = 60_000.0 / rr_intervals

        heart_rate_mean_bpm = float(np.mean(heart_rates))
        heart_rate_min_bpm = float(np.min(heart_rates))
        heart_rate_max_bpm = float(np.max(heart_rates))
        mean_rr_ms = float(np.mean(rr_intervals))

        if rr_intervals.size < 2:
            return RhythmFeatures(
                heart_rate_mean_bpm=heart_rate_mean_bpm,
                heart_rate_min_bpm=heart_rate_min_bpm,
                heart_rate_max_bpm=heart_rate_max_bpm,
                mean_rr_ms=mean_rr_ms,
                sdnn_ms=None,
                rmssd_ms=None,
                pnn50_percent=None,
                irregularity_score=None,
            )

        successive_differences = np.diff(rr_intervals)

        sdnn_ms = float(
            np.std(
                rr_intervals,
                ddof=1,
            )
        )

        rmssd_ms = float(
            np.sqrt(
                np.mean(
                    np.square(successive_differences)
                )
            )
        )

        pnn50_percent = float(
            np.mean(
                np.abs(successive_differences) > 50.0
            )
            * 100.0
        )

        irregularity_score = self._calculate_irregularity_score(
            rr_intervals=rr_intervals,
            sdnn_ms=sdnn_ms,
            rmssd_ms=rmssd_ms,
            pnn50_percent=pnn50_percent,
        )

        return RhythmFeatures(
            heart_rate_mean_bpm=heart_rate_mean_bpm,
            heart_rate_min_bpm=heart_rate_min_bpm,
            heart_rate_max_bpm=heart_rate_max_bpm,
            mean_rr_ms=mean_rr_ms,
            sdnn_ms=sdnn_ms,
            rmssd_ms=rmssd_ms,
            pnn50_percent=pnn50_percent,
            irregularity_score=irregularity_score,
        )

    def _calculate_irregularity_score(
        self,
        *,
        rr_intervals: FloatArray,
        sdnn_ms: float,
        rmssd_ms: float,
        pnn50_percent: float,
    ) -> float:
        """
        Combine three transparent rhythm-variability indicators.

        Components:

        1. RR coefficient of variation
        2. RMSSD relative to mean RR
        3. Percentage of successive RR differences above 50 ms

        Each component is normalized to the range zero to one.
        """
        mean_rr_ms = float(np.mean(rr_intervals))

        if mean_rr_ms <= 0:
            raise FeatureExtractionError(
                "Mean RR interval must be greater than zero."
            )

        coefficient_of_variation = sdnn_ms / mean_rr_ms
        rmssd_ratio = rmssd_ms / mean_rr_ms

        cv_component = float(
            np.clip(
                coefficient_of_variation / self.cv_reference,
                0.0,
                1.0,
            )
        )

        rmssd_component = float(
            np.clip(
                rmssd_ratio / self.rmssd_reference_ratio,
                0.0,
                1.0,
            )
        )

        pnn50_component = float(
            np.clip(
                pnn50_percent
                / self.pnn50_reference_percent,
                0.0,
                1.0,
            )
        )

        score = (
            0.40 * cv_component
            + 0.40 * rmssd_component
            + 0.20 * pnn50_component
        )

        return float(np.clip(score, 0.0, 1.0))

    def _validate_r_peak_series(
        self,
        *,
        r_peaks: RPeakSeries,
        rr_intervals: FloatArray,
    ) -> None:
        expected_rr_count = max(
            r_peaks.peak_count - 1,
            0,
        )

        if rr_intervals.size != expected_rr_count:
            raise FeatureExtractionError(
                "RR interval count must equal R-peak count minus one."
            )

        if rr_intervals.ndim != 1:
            raise FeatureExtractionError(
                "RR intervals must be one-dimensional."
            )

        if not np.all(np.isfinite(rr_intervals)):
            raise FeatureExtractionError(
                "RR intervals contain non-finite values."
            )

        if np.any(rr_intervals <= 0):
            raise FeatureExtractionError(
                "RR intervals must be greater than zero."
            )

        if rr_intervals.size == 0:
            return

        if np.any(rr_intervals < self.minimum_rr_ms):
            raise FeatureExtractionError(
                "An RR interval is below the configured physiological "
                "minimum."
            )

        if np.any(rr_intervals > self.maximum_rr_ms):
            raise FeatureExtractionError(
                "An RR interval exceeds the configured physiological "
                "maximum."
            )

        sample_indices = np.asarray(
            r_peaks.sample_indices,
            dtype=np.int64,
        )

        if sample_indices.size > 1:
            if np.any(np.diff(sample_indices) <= 0):
                raise FeatureExtractionError(
                    "R-peak sample indices must be strictly increasing."
                )

        timestamps = np.asarray(
            r_peaks.timestamps_seconds,
            dtype=np.float64,
        )

        if not np.all(np.isfinite(timestamps)):
            raise FeatureExtractionError(
                "R-peak timestamps contain non-finite values."
            )

        if timestamps.size > 1:
            if np.any(np.diff(timestamps) <= 0):
                raise FeatureExtractionError(
                    "R-peak timestamps must be strictly increasing."
                )

    @staticmethod
    def _empty_features() -> RhythmFeatures:
        return RhythmFeatures(
            heart_rate_mean_bpm=None,
            heart_rate_min_bpm=None,
            heart_rate_max_bpm=None,
            mean_rr_ms=None,
            sdnn_ms=None,
            rmssd_ms=None,
            pnn50_percent=None,
            irregularity_score=None,
        )

    @staticmethod
    def _validate_configuration(
        *,
        minimum_rr_ms: float,
        maximum_rr_ms: float,
        cv_reference: float,
        rmssd_reference_ratio: float,
        pnn50_reference_percent: float,
    ) -> None:
        values = (
            minimum_rr_ms,
            maximum_rr_ms,
            cv_reference,
            rmssd_reference_ratio,
            pnn50_reference_percent,
        )

        if not all(np.isfinite(value) for value in values):
            raise ValueError(
                "Rhythm extractor configuration must be finite."
            )

        if minimum_rr_ms <= 0:
            raise ValueError(
                "minimum_rr_ms must be greater than zero"
            )

        if maximum_rr_ms <= minimum_rr_ms:
            raise ValueError(
                "maximum_rr_ms must be greater than minimum_rr_ms"
            )

        if cv_reference <= 0:
            raise ValueError(
                "cv_reference must be greater than zero"
            )

        if rmssd_reference_ratio <= 0:
            raise ValueError(
                "rmssd_reference_ratio must be greater than zero"
            )

        if pnn50_reference_percent <= 0:
            raise ValueError(
                "pnn50_reference_percent must be greater than zero"
            )