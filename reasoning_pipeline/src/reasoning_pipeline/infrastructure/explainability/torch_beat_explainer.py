from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

from reasoning_pipeline.application.ports.beat_explainer import LocalAttribution
from reasoning_pipeline.infrastructure.model_lock import shared_model_lock


class TorchBeatExplainer(ABC):
    """Reusable base for gradient explainers sharing one loaded PyTorch model."""

    INPUT_LENGTH = 216

    def __init__(self, *, model: nn.Module) -> None:
        self.model = model
        self._model_lock = shared_model_lock(model)

    def _prepare_input(self, samples: tuple[float, ...]) -> torch.Tensor:
        if len(samples) != self.INPUT_LENGTH:
            raise ValueError(
                f"Expected {self.INPUT_LENGTH} samples, received {len(samples)}"
            )

        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        tensor = torch.tensor(
            samples,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("Explainer input contains non-finite values")

        return tensor

    @property
    @abstractmethod
    def method_id(self) -> str:
        ...

    @abstractmethod
    def explain(
        self,
        *,
        samples: tuple[float, ...],
        target_class: int,
    ) -> LocalAttribution | None:
        ...
