from dataclasses import dataclass

from reasoning_pipeline.domain.models.morphology_features import (
    MorphologyFeatures,
)
from reasoning_pipeline.domain.models.r_peak_series import RPeakSeries
from reasoning_pipeline.domain.models.rhythm_features import RhythmFeatures
from reasoning_pipeline.domain.models.signal_quality import SignalQuality


@dataclass(frozen=True)
class FeatureSet:
    signal_quality: SignalQuality
    r_peaks: RPeakSeries
    rhythm: RhythmFeatures
    morphology: MorphologyFeatures
    extraction_version: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.extraction_version.strip():
            raise ValueError("extraction_version cannot be empty")
