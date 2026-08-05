class BaselineAdapterError(RuntimeError):
    """Base exception for baseline inference failures."""


class CheckpointNotFoundError(BaselineAdapterError):
    """Raised when the configured checkpoint cannot be found."""


class InvalidCheckpointError(BaselineAdapterError):
    """Raised when checkpoint contents are invalid or incompatible."""


class InvalidBeatError(BaselineAdapterError):
    """Raised when an ECG beat cannot be used for inference."""
