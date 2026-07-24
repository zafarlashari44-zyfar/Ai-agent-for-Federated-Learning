class ReasoningPipelineError(Exception):
    """Base exception for the reasoning pipeline."""


class InvalidSignalError(ReasoningPipelineError):
    """Raised when an ECG signal is invalid."""


class UnsupportedSamplingRateError(ReasoningPipelineError):
    """Raised when the sampling rate is unsupported."""


class FeatureExtractionError(ReasoningPipelineError):
    """Raised when ECG feature extraction fails."""


class InsufficientSignalQualityError(ReasoningPipelineError):
    """Raised when signal quality is too poor for analysis."""


class CheckpointCompatibilityError(ReasoningPipelineError):
    """Raised when an FL checkpoint is incompatible."""
