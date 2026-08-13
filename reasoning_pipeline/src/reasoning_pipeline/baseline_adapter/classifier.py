from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from reasoning_pipeline.baseline_adapter.checkpoint import (
    calculate_sha256,
    load_checkpoint,
)
from reasoning_pipeline.baseline_adapter.cnn1d import create_model
from reasoning_pipeline.baseline_adapter.exceptions import (
    InvalidBeatError,
)
from reasoning_pipeline.baseline_adapter.labels import (
    get_class_label,
)
from reasoning_pipeline.domain.models.model_prediction import (
    ModelPrediction,
)
from reasoning_pipeline.infrastructure.model_lock import shared_model_lock


class BaselineClassifier:
    INPUT_LENGTH = 216
    NUM_CLASSES = 5
    MODEL_VERSION = "fedavg-round-10-v1"
    PREPROCESSING_VERSION = "scribe-v1-neurokit"

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = (
            Path(checkpoint_path).expanduser().resolve()
        )
        self.device = torch.device(device)

        self.model = create_model(
            input_length=self.INPUT_LENGTH,
            num_classes=self.NUM_CLASSES,
        )

        self.checkpoint = load_checkpoint(
            model=self.model,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )

        self.checkpoint_hash = calculate_sha256(
            self.checkpoint_path
        )
        self._model_lock = shared_model_lock(self.model)

    def _prepare_beat(
        self,
        beat: Sequence[float] | NDArray[np.float32],
    ) -> torch.Tensor:
        array = np.asarray(
            beat,
            dtype=np.float32,
        )

        if array.ndim != 1:
            raise InvalidBeatError(
                "A single ECG beat must be one-dimensional."
            )

        if array.shape[0] != self.INPUT_LENGTH:
            raise InvalidBeatError(
                f"Expected {self.INPUT_LENGTH} samples, "
                f"received {array.shape[0]}."
            )

        if not np.all(np.isfinite(array)):
            raise InvalidBeatError(
                "ECG beat contains NaN or infinite values."
            )

        tensor = torch.from_numpy(
            array.copy()
        ).unsqueeze(0)

        return tensor.to(
            self.device,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def predict(
        self,
        beat: Sequence[float] | NDArray[np.float32],
    ) -> ModelPrediction:
        features = self._prepare_beat(beat)

        with self._model_lock:
            self.model.eval()
            logits = self.model(features)

        probabilities_tensor = torch.softmax(
            logits,
            dim=1,
        )

        probabilities = tuple(
            float(value)
            for value in probabilities_tensor[
                0
            ].detach().cpu().tolist()
        )

        predicted_class = int(
            torch.argmax(
                probabilities_tensor,
                dim=1,
            ).item()
        )

        return ModelPrediction(
            predicted_class=predicted_class,
            predicted_label=get_class_label(
                predicted_class
            ),
            probabilities=probabilities,
            confidence=probabilities[predicted_class],
            checkpoint_path=str(self.checkpoint_path),
            checkpoint_hash=self.checkpoint_hash,
            model_version=self.MODEL_VERSION,
            preprocessing_version=(
                self.PREPROCESSING_VERSION
            ),
        )

    @torch.no_grad()
    def predict_many(
        self,
        beats: Sequence[Sequence[float] | NDArray[np.float32]],
    ) -> tuple[ModelPrediction, ...]:
        """Predict every supplied beat in one model forward pass."""
        if not beats:
            return ()

        tensors = [
            self._prepare_beat(beat).squeeze(0)
            for beat in beats
        ]
        features = torch.stack(tensors, dim=0)

        with self._model_lock:
            self.model.eval()
            logits = self.model(features)
        probabilities_tensor = torch.softmax(logits, dim=1)
        predicted_classes = torch.argmax(probabilities_tensor, dim=1)

        probability_rows = probabilities_tensor.detach().cpu().tolist()
        class_indices = predicted_classes.detach().cpu().tolist()

        return tuple(
            self._create_prediction(
                predicted_class=int(class_index),
                probabilities=tuple(float(value) for value in probability_row),
            )
            for class_index, probability_row in zip(
                class_indices,
                probability_rows,
                strict=True,
            )
        )

    def _create_prediction(
        self,
        *,
        predicted_class: int,
        probabilities: tuple[float, ...],
    ) -> ModelPrediction:
        return ModelPrediction(
            predicted_class=predicted_class,
            predicted_label=get_class_label(predicted_class),
            probabilities=probabilities,
            confidence=probabilities[predicted_class],
            checkpoint_path=str(self.checkpoint_path),
            checkpoint_hash=self.checkpoint_hash,
            model_version=self.MODEL_VERSION,
            preprocessing_version=self.PREPROCESSING_VERSION,
        )
