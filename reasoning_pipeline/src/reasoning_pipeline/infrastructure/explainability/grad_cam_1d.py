from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional
from torch.utils.hooks import RemovableHandle

from reasoning_pipeline.application.ports.beat_explainer import LocalAttribution
from reasoning_pipeline.infrastructure.explainability.torch_beat_explainer import (
    TorchBeatExplainer,
)


class GradCAM1D(TorchBeatExplainer):
    """One-dimensional Grad-CAM over a supplied convolutional target layer."""

    METHOD_ID = "grad-cam-1d"
    METHOD_VERSION = "1.0.0"
    OUTPUT_LENGTH = 216
    EXPECTED_NATIVE_RESOLUTION = 54
    INTERPOLATION_METHOD = "linear-align-corners-false"
    NORMALISATION = "relu-min-max"

    def __init__(
        self,
        *,
        model: nn.Module,
        target_layer: nn.Module,
        target_layer_name: str,
    ) -> None:
        super().__init__(model=model)
        if not target_layer_name.strip():
            raise ValueError("target_layer_name cannot be empty")

        self.target_layer = target_layer
        self.target_layer_name = target_layer_name

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
            raise ValueError("target_class must be between 0 and 4")

        with self._model_lock:
            return self._explain_locked(
                samples=samples,
                target_class=target_class,
            )

    def _explain_locked(
        self,
        *,
        samples: tuple[float, ...],
        target_class: int,
    ) -> LocalAttribution:
        activations: torch.Tensor | None = None
        gradients: torch.Tensor | None = None

        def capture_activations(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            nonlocal activations
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(
                    "Grad-CAM target layer must return a tensor"
                )
            activations = output.detach()

        def capture_gradients(
            _module: nn.Module,
            _gradient_inputs: tuple[torch.Tensor, ...] | torch.Tensor,
            gradient_outputs: tuple[torch.Tensor, ...] | torch.Tensor,
        ) -> tuple[torch.Tensor, ...] | torch.Tensor | None:
            nonlocal gradients
            output_gradient = (
                gradient_outputs
                if isinstance(gradient_outputs, torch.Tensor)
                else gradient_outputs[0] if gradient_outputs else None
            )
            if output_gradient is None:
                raise RuntimeError(
                    "Grad-CAM target layer did not produce gradients"
                )
            gradients = output_gradient.detach()
            return None

        handles: tuple[RemovableHandle, ...] = (
            self.target_layer.register_forward_hook(capture_activations),
            self.target_layer.register_full_backward_hook(capture_gradients),
        )
        was_training = self.model.training

        try:
            self.model.eval()
            input_tensor = self._prepare_input(samples)
            self.model.zero_grad(set_to_none=True)

            logits = self.model(input_tensor)
            if logits.ndim != 2 or logits.shape[0] != 1:
                raise RuntimeError(
                    "Grad-CAM model must return one row of class logits"
                )
            if target_class >= logits.shape[1]:
                raise ValueError(
                    f"target_class {target_class} is outside model output"
                )

            target_logit = logits[0, target_class]
            target_logit.backward()

            if activations is None or gradients is None:
                raise RuntimeError(
                    "Grad-CAM hooks did not capture activations and gradients"
                )
            if activations.shape != gradients.shape:
                raise RuntimeError(
                    "Grad-CAM activation and gradient shapes must match"
                )
            if activations.ndim != 3 or activations.shape[0] != 1:
                raise RuntimeError(
                    "Grad-CAM target must produce batch-channel-time data"
                )

            native_resolution = int(activations.shape[-1])
            if native_resolution != self.EXPECTED_NATIVE_RESOLUTION:
                raise RuntimeError(
                    "Unexpected Grad-CAM native resolution: "
                    f"{native_resolution}"
                )

            channel_weights = gradients.mean(dim=2, keepdim=True)
            native_map = torch.relu(
                (channel_weights * activations).sum(dim=1, keepdim=True)
            )
            interpolated = functional.interpolate(
                native_map,
                size=self.OUTPUT_LENGTH,
                mode="linear",
                align_corners=False,
            ).flatten()

            minimum = interpolated.min()
            shifted = interpolated - minimum
            maximum = shifted.max()
            if float(maximum) > torch.finfo(shifted.dtype).eps:
                normalised = shifted / maximum
            else:
                normalised = torch.zeros_like(shifted)

            if not bool(torch.isfinite(normalised).all()):
                raise RuntimeError("Grad-CAM produced non-finite attribution")

            values = tuple(
                float(value)
                for value in normalised.detach().cpu().tolist()
            )

            return LocalAttribution(
                method_id=self.method_id,
                method_version=self.METHOD_VERSION,
                target_class=target_class,
                target_output="logit",
                values=values,
                signed=False,
                native_resolution=native_resolution,
                interpolation_method=self.INTERPOLATION_METHOD,
                normalisation=self.NORMALISATION,
                parameters=(
                    ("target_layer", self.target_layer_name),
                    ("relu", "true"),
                ),
            )
        finally:
            for handle in handles:
                handle.remove()
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)
