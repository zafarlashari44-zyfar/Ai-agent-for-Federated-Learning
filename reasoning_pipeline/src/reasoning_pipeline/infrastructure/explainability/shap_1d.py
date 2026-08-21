from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import shap
import torch
from torch import nn

from reasoning_pipeline.application.ports.beat_explainer import LocalAttribution
from reasoning_pipeline.infrastructure.explainability.torch_beat_explainer import (
    TorchBeatExplainer,
)


class SHAP1D(TorchBeatExplainer):
    """Gradient SHAP attribution for one prepared 216 sample ECG beat."""

    METHOD_ID = "shap-gradient-1d"
    METHOD_VERSION = "1.0.0"
    INPUT_LENGTH = 216

    def __init__(
        self,
        *,
        model: nn.Module,
        background_path: str | Path,
    ) -> None:
        super().__init__(model=model)

        self.background_path = Path(background_path)

        if not self.background_path.exists():
            raise FileNotFoundError(
                f"SHAP background not found: {self.background_path}"
            )

        background_array = np.load(
            self.background_path
        ).astype(np.float32)

        if background_array.ndim != 2:
            raise ValueError(
                "SHAP background must be two dimensional"
            )

        if background_array.shape[1] != self.INPUT_LENGTH:
            raise ValueError(
                f"Expected SHAP background length {self.INPUT_LENGTH}"
            )

        if not np.all(np.isfinite(background_array)):
            raise ValueError(
                "SHAP background contains non finite values"
            )

        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        self.background = torch.tensor(
            background_array,
            dtype=torch.float32,
            device=device,
        )

        self.explainer = shap.GradientExplainer(
            self.model,
            self.background,
        )

    @property
    def method_id(self) -> str:
        return self.METHOD_ID

    def explain(
        self,
        *,
        samples: tuple[float, ...],
        target_class: int,
    ) -> LocalAttribution:
        if not 0 <= target_class < 5:
            raise ValueError(
                "target_class must be between 0 and 4"
            )

        with self._model_lock:
            self.model.eval()

            input_tensor = self._prepare_input(
                samples
            )

            shap_values = self.explainer.shap_values(
                input_tensor
            )

        values = np.asarray(shap_values)

        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]

        if values.ndim == 2:
            if values.shape[1] == 5:
                attribution = values[:, target_class]
            elif values.shape[0] == 5:
                attribution = values[target_class]
            else:
                raise RuntimeError(
                    f"Unexpected SHAP output shape {values.shape}"
                )
        elif values.ndim == 1:
            attribution = values
        else:
            raise RuntimeError(
                f"Unexpected SHAP output shape {values.shape}"
            )

        attribution = np.asarray(
            attribution,
            dtype=np.float32,
        ).reshape(-1)

        if attribution.shape != (self.INPUT_LENGTH,):
            raise RuntimeError(
                "SHAP attribution must contain exactly 216 values"
            )

        if not np.all(np.isfinite(attribution)):
            raise RuntimeError(
                "SHAP produced non finite attribution"
            )

        return LocalAttribution(
            method_id=self.method_id,
            method_version=self.METHOD_VERSION,
            target_class=target_class,
            target_output="logit",
            values=tuple(
                float(value)
                for value in attribution.tolist()
            ),
            signed=True,
            native_resolution=self.INPUT_LENGTH,
            interpolation_method=None,
            normalisation="none",
            parameters=(
                (
                    "background_size",
                    str(self.background.shape[0]),
                ),
                (
                    "background_source",
                    self.background_path.name,
                ),
            ),
        )
