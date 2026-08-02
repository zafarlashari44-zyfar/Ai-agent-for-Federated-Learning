from typing import Protocol

from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.domain.models.model_prediction import ModelPrediction
from reasoning_pipeline.domain.models.ood_assessment import OODAssessment
from reasoning_pipeline.domain.models.signal_suitability_assessment import (
    SignalSuitabilityAssessment,
)


class OODAssessorProtocol(Protocol):
    def assess(
        self,
        *,
        signal: ECGSignal,
        predictions: tuple[ModelPrediction, ...],
        suitability: SignalSuitabilityAssessment,
    ) -> OODAssessment:
        ...
