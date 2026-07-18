from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import shap
import torch
from torch import nn

from fl_ecg_orchestrator.data.loader import (
    CLASS_NAMES,
    load_config,
    load_global_test_data,
    resolve_project_path,
)
from fl_ecg_orchestrator.model.cnn1d import create_model


class ProbabilityModel(nn.Module):
    """Wrap a logits model so SHAP explains class probabilities."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(x), dim=1)


class ECGSHAPExplainer:
    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path = "fl_ecg_orchestrator/config/config.yaml",
        background_size: int = 100,
        explain_size: int = 250,
        seed: int | None = None,
        device: str | None = None,
    ) -> None:
        self.config_path = str(config_path)
        self.config = load_config(config_path)
        self.test_data = load_global_test_data(config_path)

        self.input_length = int(self.config["data"]["input_length"])
        self.num_classes = int(self.config["data"]["num_classes"])
        self.seed = (
            int(seed)
            if seed is not None
            else int(self.config["project"]["seed"])
        )
        self.background_size = int(background_size)
        self.explain_size = int(explain_size)

        if self.background_size <= 0:
            raise ValueError("background_size must be greater than zero")
        if self.explain_size <= 0:
            raise ValueError("explain_size must be greater than zero")

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.checkpoint_path = resolve_project_path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {self.checkpoint_path}"
            )

        self.model = create_model(
            input_length=self.input_length,
            num_classes=self.num_classes,
        ).to(self.device)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        self.probability_model = ProbabilityModel(self.model).to(self.device)
        self.probability_model.eval()

        self.features = np.asarray(
            self.test_data["features"],
            dtype=np.float32,
        )
        self.labels = np.asarray(
            self.test_data["labels"],
            dtype=np.int64,
        )

        if self.features.ndim != 2:
            raise ValueError(
                f"Expected global test features with shape (n, length), "
                f"received {self.features.shape}"
            )

        if self.features.shape[1] != self.input_length:
            raise ValueError(
                f"Expected input length {self.input_length}, "
                f"received {self.features.shape[1]}"
            )

        self.rng = np.random.default_rng(self.seed)
        self.background_indices = self._stratified_indices(
            self.labels,
            min(self.background_size, len(self.labels)),
        )
        self.explain_indices = self._stratified_indices(
            self.labels,
            min(self.explain_size, len(self.labels)),
        )

        self.background = torch.from_numpy(
            self.features[self.background_indices]
        ).to(self.device)

        self.explain_tensor = torch.from_numpy(
            self.features[self.explain_indices]
        ).to(self.device)

        self.explainer_name = ""
        self.explainer = self._build_explainer()

    def _stratified_indices(
        self,
        labels: np.ndarray,
        sample_size: int,
    ) -> np.ndarray:
        if sample_size >= len(labels):
            return np.arange(len(labels), dtype=np.int64)

        selected: list[int] = []
        unique_classes = np.unique(labels)

        per_class = max(1, sample_size // max(len(unique_classes), 1))

        for class_id in unique_classes:
            class_indices = np.flatnonzero(labels == class_id)
            take = min(per_class, len(class_indices))
            if take > 0:
                chosen = self.rng.choice(
                    class_indices,
                    size=take,
                    replace=False,
                )
                selected.extend(int(i) for i in chosen)

        selected = list(dict.fromkeys(selected))

        if len(selected) < sample_size:
            remaining = np.setdiff1d(
                np.arange(len(labels)),
                np.asarray(selected, dtype=np.int64),
                assume_unique=False,
            )
            extra = self.rng.choice(
                remaining,
                size=sample_size - len(selected),
                replace=False,
            )
            selected.extend(int(i) for i in extra)

        self.rng.shuffle(selected)
        return np.asarray(selected[:sample_size], dtype=np.int64)

    def _build_explainer(self) -> Any:
        try:
            explainer = shap.DeepExplainer(
                self.probability_model,
                self.background,
            )
            self.explainer_name = "DeepExplainer"
            return explainer
        except Exception:
            explainer = shap.GradientExplainer(
                self.probability_model,
                self.background,
            )
            self.explainer_name = "GradientExplainer"
            return explainer

    @torch.no_grad()
    def predict(
        self,
        tensor: torch.Tensor | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if tensor is None:
            tensor = self.explain_tensor

        probabilities = self.probability_model(tensor).cpu().numpy()
        predictions = probabilities.argmax(axis=1).astype(np.int64)
        return probabilities, predictions

    def compute_shap_values(self) -> np.ndarray:
        raw_values = self.explainer.shap_values(
            self.explain_tensor,
            check_additivity=False,
        )

        values = self._normalise_shap_output(raw_values)
        return values.astype(np.float32, copy=False)

    def _normalise_shap_output(self, raw_values: Any) -> np.ndarray:
        if isinstance(raw_values, list):
            arrays = [np.asarray(value) for value in raw_values]
            values = np.stack(arrays, axis=1)
        else:
            values = np.asarray(raw_values)

        # Desired shape: (samples, classes, features)
        if values.ndim == 4 and values.shape[2] == 1:
            values = values.squeeze(2)

        if values.ndim != 3:
            raise ValueError(
                f"Unsupported SHAP output shape {values.shape}"
            )

        n_samples = len(self.explain_indices)

        if values.shape[0] == n_samples and values.shape[1] == self.num_classes:
            return values

        if values.shape[0] == n_samples and values.shape[2] == self.num_classes:
            return np.transpose(values, (0, 2, 1))

        if values.shape[0] == self.num_classes and values.shape[1] == n_samples:
            return np.transpose(values, (1, 0, 2))

        raise ValueError(
            f"Could not convert SHAP output shape {values.shape} "
            f"to (samples, classes, features)"
        )

    def run(self) -> dict[str, Any]:
        shap_values = self.compute_shap_values()
        probabilities, predictions = self.predict()

        predicted_values = np.stack(
            [
                shap_values[row_index, class_id]
                for row_index, class_id in enumerate(predictions)
            ],
            axis=0,
        )

        return {
            "shap_values": shap_values,
            "predicted_shap_values": predicted_values,
            "features": self.features[self.explain_indices],
            "true_labels": self.labels[self.explain_indices],
            "predictions": predictions,
            "probabilities": probabilities,
            "global_indices": self.explain_indices,
            "background_indices": self.background_indices,
            "explainer": self.explainer_name,
        }

    def save_raw_results(
        self,
        output_dir: str | Path,
        results: dict[str, Any],
    ) -> Path:
        output_path = resolve_project_path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_path / "shap_results.npz",
            shap_values=results["shap_values"],
            predicted_shap_values=results["predicted_shap_values"],
            features=results["features"],
            true_labels=results["true_labels"],
            predictions=results["predictions"],
            probabilities=results["probabilities"],
            global_indices=results["global_indices"],
            background_indices=results["background_indices"],
        )

        metadata = {
            "checkpoint": str(self.checkpoint_path),
            "config_path": self.config_path,
            "device": str(self.device),
            "explainer": results["explainer"],
            "input_length": self.input_length,
            "num_classes": self.num_classes,
            "background_size": int(len(self.background_indices)),
            "explained_samples": int(len(self.explain_indices)),
            "class_names": {
                str(key): value for key, value in CLASS_NAMES.items()
            },
        }

        with (output_path / "metadata.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metadata, file, indent=2)

        return output_path
