class NarrativeGenerationError(RuntimeError):
    """Raised when narrative generation fails without fallback."""


class NarrativeValidationError(ValueError):
    """Raised when generated output violates the narrative contract."""
