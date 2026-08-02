class ReasoningPipelineError(Exception):
    """Base exception for the reasoning pipeline."""


class InvalidSignalError(ReasoningPipelineError):
    """Raised when an ECG signal is invalid."""


class UnsupportedSamplingRateError(ReasoningPipelineError):
    """Raised when the sampling rate is unsupported."""


class SignalSuitabilityRejectedError(ReasoningPipelineError):
    """Raised when an ECG fails the pre-inference technical gate."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("Signal suitability rejected: " + "; ".join(reasons))


class FeatureExtractionError(ReasoningPipelineError):
    """Raised when ECG feature extraction fails."""


class InsufficientSignalQualityError(ReasoningPipelineError):
    """Raised when signal quality is too poor for analysis."""


class CheckpointCompatibilityError(ReasoningPipelineError):
    """Raised when an FL checkpoint is incompatible."""
