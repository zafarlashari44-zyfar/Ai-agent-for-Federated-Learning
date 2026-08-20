from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import shap
import torch
from torch import nn


class SHAP1D:
    METHOD_ID = "shap-gradient-1d"
    METHOD_VERSION = "1.0.0"
    INPUT_LENGTH = 216

    def __init__(
        self,
        *,
        model: nn.Module,
        background: Sequence[Sequence[float]],
    ) -> None:
        self.model = model

        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

        background_array = np.asarray(
            background,
            dtype=np.float32,
        )

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

        self.background = torch.tensor(
            background_array,
            dtype=torch.float32,
            device=self.device,
        )

        self.model.eval()

        self.explainer = shap.GradientExplainer(
            self.model,
            self.background,
        )

    def explain(
        self,
        *,
        samples: Sequence[float],
        target_class: int,
    ) -> np.ndarray:

        samples_array = np.asarray(
            samples,
            dtype=np.float32,
        )

        if samples_array.shape != (self.INPUT_LENGTH,):
            raise ValueError(
                f"Expected {self.INPUT_LENGTH} samples"
            )

        if not np.all(np.isfinite(samples_array)):
            raise ValueError(
                "SHAP input contains non finite values"
            )

        if not 0 <= target_class < 5:
            raise ValueError(
                "target_class must be between 0 and 4"
            )

        input_tensor = torch.tensor(
            samples_array,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        shap_values = self.explainer.shap_values(
            input_tensor,
        )

        values = np.asarray(shap_values)

        if values.ndim == 3:
            if values.shape[0] == 1:
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
                f"Expected 216 SHAP values, received {attribution.shape}"
            )

        if not np.all(np.isfinite(attribution)):
            raise RuntimeError(
                "SHAP produced non finite attribution"
            )

        return attribution