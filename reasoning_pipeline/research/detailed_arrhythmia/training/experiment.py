from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader

from research.detailed_arrhythmia import MODEL_VERSION
from research.detailed_arrhythmia.config import TrainingConfig
from research.detailed_arrhythmia.dataset.ontology import DetailedLabelOntology
from research.detailed_arrhythmia.dataset.splits import SplitManifest
from research.detailed_arrhythmia.evaluation.metrics import calculate_metrics
from research.detailed_arrhythmia.training.augmentation import SafeECGAugmenter
from research.detailed_arrhythmia.training.dataset import BeatDataset
from research.detailed_arrhythmia.training.metadata import (
    CheckpointMetadata,
    sha256_file,
)
from research.detailed_arrhythmia.training.model import create_detailed_model
from research.detailed_arrhythmia.training.weights import calculate_class_weights


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=True)


def run_experiment(
    *,
    train_beats: NDArray[np.float32],
    train_labels: NDArray[np.int64],
    validation_beats: NDArray[np.float32],
    validation_labels: NDArray[np.int64],
    test_beats: NDArray[np.float32],
    test_labels: NDArray[np.int64],
    ontology: DetailedLabelOntology,
    split_manifest: SplitManifest,
    config: TrainingConfig,
    augmentation_enabled: bool,
    output_dir: Path,
) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = calculate_class_weights(
        train_labels,
        len(ontology.labels),
        method=config.class_weight_method,  # type: ignore[arg-type]
        effective_number_beta=config.effective_number_beta,
    )
    transform = (
        SafeECGAugmenter(config.augmentation, seed=config.seed)
        if augmentation_enabled
        else None
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        BeatDataset(train_beats, train_labels, transform=transform),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        BeatDataset(validation_beats, validation_labels),
        batch_size=config.batch_size,
    )
    test_loader = DataLoader(
        BeatDataset(test_beats, test_labels),
        batch_size=config.batch_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_detailed_model(len(ontology.labels)).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_macro_f1 = -1.0
    patience = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for beats, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(beats.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        validation_metrics = evaluate(model, validation_loader, ontology.labels, device)
        history.append(
            {
                "epoch": float(epoch),
                "training_loss": float(np.mean(losses)),
                "validation_macro_f1": validation_metrics["macro_f1"],
            }
        )
        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path = output_dir / "detailed_classifier.pt"
    torch.save(best_state, checkpoint_path)
    validation_metrics = evaluate(model, validation_loader, ontology.labels, device)
    test_metrics = evaluate(model, test_loader, ontology.labels, device)
    metadata = CheckpointMetadata(
        model_version=MODEL_VERSION,
        ontology_version=ontology.version,
        labels=ontology.labels,
        split_manifest_hash=split_manifest.sha256,
        training_seed=config.seed,
        class_weight_method=config.class_weight_method,
        class_weights=tuple(float(value) for value in weights),
        augmentation_configuration=(
            asdict(config.augmentation) if augmentation_enabled else None
        ),
        checkpoint_hash=sha256_file(checkpoint_path),
    )
    metadata.write(output_dir / "checkpoint_metadata.json")
    result = {
        "configuration": config.to_dict(),
        "augmentation_enabled": augmentation_enabled,
        "history": history,
        "validation": validation_metrics,
        "test": test_metrics,
        "checkpoint_metadata": asdict(metadata),
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    labels: tuple[str, ...],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    with torch.no_grad():
        for beats, batch_targets in loader:
            logits = model(beats.to(device))
            batch_probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            probabilities.extend(batch_probabilities.tolist())
            predictions.extend(np.argmax(batch_probabilities, axis=1).tolist())
            targets.extend(batch_targets.numpy().tolist())
    return calculate_metrics(
        np.asarray(targets, dtype=np.int64),
        np.asarray(predictions, dtype=np.int64),
        np.asarray(probabilities, dtype=np.float64),
        labels,
    )
