import random
import time
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader


CLASS_NAMES = [
    "N",
    "S",
    "V",
    "F",
    "Q",
]


def set_reproducibility(
    seed: int,
) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )


def get_device() -> torch.device:

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def snapshot_global_parameters(
    model: nn.Module,
) -> dict[str, torch.Tensor]:

    return {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }


def calculate_proximal_penalty(
    model: nn.Module,
    global_parameters: dict[str, torch.Tensor],
) -> torch.Tensor:

    penalty = torch.zeros(
        1,
        device=next(model.parameters()).device,
    )

    for name, parameter in model.named_parameters():

        reference = global_parameters[name].to(
            parameter.device
        )

        penalty = penalty + torch.sum(
            torch.square(
                parameter - reference
            )
        )

    return penalty.squeeze()


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | None = None,
) -> dict[str, Any]:

    device = device or get_device()

    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    sample_count = 0

    predictions = []
    targets = []

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            logits = model(features)

            loss = criterion(
                logits,
                labels,
            )

            batch_size = labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            sample_count += batch_size

            predicted_labels = torch.argmax(
                logits,
                dim=1,
            )

            predictions.extend(
                predicted_labels
                .cpu()
                .numpy()
                .tolist()
            )

            targets.extend(
                labels
                .cpu()
                .numpy()
                .tolist()
            )

    predictions_array = np.asarray(
        predictions,
        dtype=np.int64,
    )

    targets_array = np.asarray(
        targets,
        dtype=np.int64,
    )

    per_class_recall = recall_score(
        targets_array,
        predictions_array,
        labels=[0, 1, 2, 3, 4],
        average=None,
        zero_division=0,
    )

    per_class_f1 = f1_score(
        targets_array,
        predictions_array,
        labels=[0, 1, 2, 3, 4],
        average=None,
        zero_division=0,
    )

    matrix = confusion_matrix(
        targets_array,
        predictions_array,
        labels=[0, 1, 2, 3, 4],
    )

    return {
        "loss": (
            total_loss
            / max(sample_count, 1)
        ),
        "accuracy": accuracy_score(
            targets_array,
            predictions_array,
        ),
        "macro_precision": precision_score(
            targets_array,
            predictions_array,
            labels=[0, 1, 2, 3, 4],
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            targets_array,
            predictions_array,
            labels=[0, 1, 2, 3, 4],
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            targets_array,
            predictions_array,
            labels=[0, 1, 2, 3, 4],
            average="macro",
            zero_division=0,
        ),
        "per_class_recall": {
            class_name: float(value)
            for class_name, value
            in zip(
                CLASS_NAMES,
                per_class_recall,
            )
        },
        "per_class_f1": {
            class_name: float(value)
            for class_name, value
            in zip(
                CLASS_NAMES,
                per_class_f1,
            )
        },
        "confusion_matrix": matrix.tolist(),
        "num_samples": int(sample_count),
    }


def train_local_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    local_epochs: int,
    learning_rate: float,
    proximal_mu: float = 0.0,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict[str, Any]:

    set_reproducibility(seed)

    device = device or get_device()

    model = model.to(device)

    global_parameters = (
        snapshot_global_parameters(model)
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    start_time = time.perf_counter()

    epoch_history = []

    for epoch in range(
        1,
        local_epochs + 1,
    ):

        model.train()

        running_loss = 0.0
        correct = 0
        sample_count = 0

        for features, labels in train_loader:

            features = features.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad()

            logits = model(features)

            classification_loss = criterion(
                logits,
                labels,
            )

            proximal_penalty = (
                calculate_proximal_penalty(
                    model,
                    global_parameters,
                )
                if proximal_mu > 0
                else torch.zeros(
                    1,
                    device=device,
                ).squeeze()
            )

            total_loss = (
                classification_loss
                + 0.5
                * proximal_mu
                * proximal_penalty
            )

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            batch_size = labels.size(0)

            running_loss += (
                total_loss.item()
                * batch_size
            )

            predicted_labels = torch.argmax(
                logits,
                dim=1,
            )

            correct += (
                predicted_labels == labels
            ).sum().item()

            sample_count += batch_size

        training_loss = (
            running_loss
            / max(sample_count, 1)
        )

        training_accuracy = (
            correct
            / max(sample_count, 1)
        )

        validation_metrics = evaluate_model(
            model,
            validation_loader,
            device,
        )

        epoch_history.append(
            {
                "epoch": epoch,
                "training_loss": float(
                    training_loss
                ),
                "training_accuracy": float(
                    training_accuracy
                ),
                "validation_loss": float(
                    validation_metrics["loss"]
                ),
                "validation_accuracy": float(
                    validation_metrics["accuracy"]
                ),
                "validation_macro_f1": float(
                    validation_metrics["macro_f1"]
                ),
            }
        )

        print(
            f"Epoch {epoch}/{local_epochs} "
            f"train_loss={training_loss:.4f} "
            f"train_accuracy={training_accuracy:.4f} "
            f"validation_loss="
            f"{validation_metrics['loss']:.4f} "
            f"validation_accuracy="
            f"{validation_metrics['accuracy']:.4f} "
            f"validation_macro_f1="
            f"{validation_metrics['macro_f1']:.4f}"
        )

    duration_seconds = (
        time.perf_counter()
        - start_time
    )

    final_validation = evaluate_model(
        model,
        validation_loader,
        device,
    )

    return {
        "model": model,
        "num_examples": len(
            train_loader.dataset
        ),
        "duration_seconds": float(
            duration_seconds
        ),
        "proximal_mu": float(
            proximal_mu
        ),
        "epoch_history": epoch_history,
        "final_validation": final_validation,
    }


def model_state_to_numpy(
    model: nn.Module,
) -> list[np.ndarray]:

    return [
        tensor.detach()
        .cpu()
        .numpy()
        .copy()
        for tensor
        in model.state_dict().values()
    ]


def numpy_to_model_state(
    model: nn.Module,
    parameters: list[np.ndarray],
) -> None:

    state_dict = model.state_dict()

    if len(parameters) != len(state_dict):
        raise ValueError(
            "Received parameter list does not match "
            "the shared CNN state dictionary"
        )

    updated_state = {}

    for (
        key,
        reference_tensor,
    ), array in zip(
        state_dict.items(),
        parameters,
    ):

        tensor = torch.from_numpy(
            np.asarray(array)
        )

        if tensor.shape != reference_tensor.shape:
            raise ValueError(
                f"Parameter shape mismatch for {key}. "
                f"Expected {tuple(reference_tensor.shape)}, "
                f"received {tuple(tensor.shape)}"
            )

        updated_state[key] = tensor.to(
            dtype=reference_tensor.dtype
        )

    model.load_state_dict(
        updated_state,
        strict=True,
    )
