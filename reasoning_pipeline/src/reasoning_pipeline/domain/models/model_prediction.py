from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrediction:
    predicted_class: int
    predicted_label: str
    probabilities: tuple[float, ...]
    confidence: float
    checkpoint_path: str
    checkpoint_hash: str
    model_version: str
    preprocessing_version: str

    def __post_init__(self) -> None:
        if self.predicted_class < 0:
            raise ValueError("predicted_class cannot be negative")

        if not self.predicted_label.strip():
            raise ValueError("predicted_label cannot be empty")

        if not self.probabilities:
            raise ValueError("probabilities cannot be empty")

        if any(value < 0.0 or value > 1.0 for value in self.probabilities):
            raise ValueError(
                "probabilities must be between zero and one"
            )

        if abs(sum(self.probabilities) - 1.0) > 1e-4:
            raise ValueError("probabilities must sum to one")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
