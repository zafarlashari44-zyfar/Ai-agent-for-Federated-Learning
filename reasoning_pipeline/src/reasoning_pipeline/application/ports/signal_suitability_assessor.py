from typing import Protocol

from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.domain.models.signal_suitability_assessment import (
    SignalSuitabilityAssessment,
)


class SignalSuitabilityAssessorProtocol(Protocol):
    def assess(self, signal: ECGSignal) -> SignalSuitabilityAssessment:
        ...
