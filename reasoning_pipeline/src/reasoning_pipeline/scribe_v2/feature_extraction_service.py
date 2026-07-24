from __future__ import annotations

from typing import Protocol

from reasoning_pipeline.domain.enums.statuses import (
    SignalQualityStatus,
)
from reasoning_pipeline.domain.exceptions.pipeline_errors import (
    InsufficientSignalQualityError,
)
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.domain.models.feature_set import FeatureSet
from reasoning_pipeline.domain.models.morphology_features import (
    MorphologyFeatures,
)
from reasoning_pipeline.domain.models.r_peak_series import RPeakSeries
from reasoning_pipeline.domain.models.rhythm_features import RhythmFeatures
from reasoning_pipeline.domain.models.signal_quality import SignalQuality
from reasoning_pipeline.scribe_v2.quality_assessor import (
    ECGSignalQualityAssessor,
)
from reasoning_pipeline.scribe_v2.r_peak_detector import (
    SciPyRPeakDetector,
)
from reasoning_pipeline.scribe_v2.rhythm_feature_extractor import (
    RhythmFeatureExtractor,
)


class SignalQualityAssessorProtocol(Protocol):
    """Contract for ECG signal-quality assessment."""

    def assess(self, signal: ECGSignal) -> SignalQuality:
        """Assess the technical quality of an ECG signal."""


class RPeakDetectorProtocol(Protocol):
    """Contract for R-peak detection."""

    def detect(
        self,
        signal: ECGSignal,
        *,
        signal_quality: SignalQuality | None = None,
    ) -> RPeakSeries:
        """Detect R peaks from an ECG signal."""


class RhythmFeatureExtractorProtocol(Protocol):
    """Contract for rhythm-feature extraction."""

    def extract(
        self,
        r_peaks: RPeakSeries,
    ) -> RhythmFeatures:
        """Extract rhythm features from an R-peak series."""


class MorphologyFeatureExtractorProtocol(Protocol):
    """Contract for future ECG morphology extraction."""

    def extract(
        self,
        signal: ECGSignal,
        r_peaks: RPeakSeries,
    ) -> MorphologyFeatures:
        """Extract morphology features from an ECG signal."""


class ScribeV2FeatureExtractionService:
    """
    Coordinate the complete Scribe v2 feature-extraction workflow.

    The service performs orchestration only. Signal processing remains
    inside the independently testable quality, R-peak, rhythm, and
    morphology components.

    Current flow:

    ECGSignal
        -> SignalQuality
        -> RPeakSeries
        -> RhythmFeatures
        -> MorphologyFeatures
        -> FeatureSet

    Morphology extraction is optional until the dedicated morphology
    extractor is implemented. When unavailable, an explicit empty
    morphology result and warning are included.
    """

    EXTRACTION_VERSION = "scribe-v2.0.0"

    MORPHOLOGY_UNAVAILABLE_WARNING = (
        "Morphology features were not extracted because the morphology "
        "extractor is not yet configured."
    )

    def __init__(
        self,
        *,
        quality_assessor: SignalQualityAssessorProtocol | None = None,
        r_peak_detector: RPeakDetectorProtocol | None = None,
        rhythm_extractor: RhythmFeatureExtractorProtocol | None = None,
        morphology_extractor: (
            MorphologyFeatureExtractorProtocol | None
        ) = None,
        extraction_version: str = EXTRACTION_VERSION,
    ) -> None:
        normalized_version = extraction_version.strip()

        if not normalized_version:
            raise ValueError("extraction_version cannot be empty")

        self.quality_assessor = (
            quality_assessor or ECGSignalQualityAssessor()
        )
        self.r_peak_detector = (
            r_peak_detector or SciPyRPeakDetector()
        )
        self.rhythm_extractor = (
            rhythm_extractor or RhythmFeatureExtractor()
        )
        self.morphology_extractor = morphology_extractor
        self.extraction_version = normalized_version

    def extract(
        self,
        signal: ECGSignal,
    ) -> FeatureSet:
        """
        Extract all currently available Scribe v2 ECG features.

        Args:
            signal:
                Validated immutable ECG signal.

        Returns:
            An immutable FeatureSet containing quality, R-peak, rhythm,
            morphology, version, and warning information.

        Raises:
            InsufficientSignalQualityError:
                If the signal-quality assessor classifies the ECG as
                unusable.
        """
        signal_quality = self.quality_assessor.assess(signal)

        self._ensure_signal_is_usable(signal_quality)

        r_peaks = self.r_peak_detector.detect(
            signal,
            signal_quality=signal_quality,
        )

        rhythm = self.rhythm_extractor.extract(r_peaks)

        morphology, morphology_warnings = (
            self._extract_morphology(
                signal=signal,
                r_peaks=r_peaks,
            )
        )

        warnings = self._merge_warnings(
            signal_quality.warnings,
            morphology_warnings,
        )

        return FeatureSet(
            signal_quality=signal_quality,
            r_peaks=r_peaks,
            rhythm=rhythm,
            morphology=morphology,
            extraction_version=self.extraction_version,
            warnings=warnings,
        )

    @staticmethod
    def _ensure_signal_is_usable(
        signal_quality: SignalQuality,
    ) -> None:
        if signal_quality.status is SignalQualityStatus.UNUSABLE:
            warning_context = " ".join(signal_quality.warnings)

            message = (
                "Scribe v2 feature extraction cannot continue because "
                "the ECG signal is unusable."
            )

            if warning_context:
                message = f"{message} {warning_context}"

            raise InsufficientSignalQualityError(message)

    def _extract_morphology(
        self,
        *,
        signal: ECGSignal,
        r_peaks: RPeakSeries,
    ) -> tuple[MorphologyFeatures, tuple[str, ...]]:
        if self.morphology_extractor is None:
            return (
                self._empty_morphology_features(),
                (self.MORPHOLOGY_UNAVAILABLE_WARNING,),
            )

        morphology = self.morphology_extractor.extract(
            signal,
            r_peaks,
        )

        return morphology, ()

    @staticmethod
    def _empty_morphology_features() -> MorphologyFeatures:
        return MorphologyFeatures(
            mean_qrs_duration_ms=None,
            mean_pr_interval_ms=None,
            mean_qt_interval_ms=None,
            mean_r_amplitude=None,
            abnormal_beat_count=None,
            morphology_confidence=0.0,
        )

    @staticmethod
    def _merge_warnings(
        *warning_groups: tuple[str, ...],
    ) -> tuple[str, ...]:
        """
        Merge warnings while preserving order and removing duplicates.
        """
        merged: list[str] = []
        observed: set[str] = set()

        for warning_group in warning_groups:
            for warning in warning_group:
                normalized_warning = warning.strip()

                if (
                    normalized_warning
                    and normalized_warning not in observed
                ):
                    merged.append(normalized_warning)
                    observed.add(normalized_warning)

        return tuple(merged)