from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from research.detailed_arrhythmia.config import TrainingConfig
from research.detailed_arrhythmia.dataset.beats import (
    extract_expert_annotated_beats,
)
from research.detailed_arrhythmia.dataset.ontology import DEFAULT_ONTOLOGY
from research.detailed_arrhythmia.evaluation.metrics import calculate_metrics
from research.detailed_arrhythmia.training.dataset import BeatDataset
from research.detailed_arrhythmia.training.experiment import set_seed
from research.detailed_arrhythmia.training.model import create_detailed_model
from research.detailed_arrhythmia.training.weights import (
    calculate_class_weights,
)

TRAINING_SEED = 42


def evaluate_fold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()

    targets: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []

    with torch.no_grad():
        for beats, batch_targets in loader:
            logits = model(beats.to(device))

            batch_probabilities = (
                torch.softmax(logits, dim=1)
                .cpu()
                .numpy()
            )

            probabilities.extend(
                batch_probabilities.tolist()
            )

            predictions.extend(
                np.argmax(
                    batch_probabilities,
                    axis=1,
                ).tolist()
            )

            targets.extend(
                batch_targets.numpy().tolist()
            )

    target_array = np.asarray(
        targets,
        dtype=np.int64,
    )

    prediction_array = np.asarray(
        predictions,
        dtype=np.int64,
    )

    probability_array = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    metrics = calculate_metrics(
        target_array,
        prediction_array,
        probability_array,
        DEFAULT_ONTOLOGY.labels,
    )

    present_labels = [
        label
        for label in DEFAULT_ONTOLOGY.labels
        if metrics["per_class"][label]["support"] > 0
    ]

    present_f1_values = [
        metrics["per_class"][label]["f1"]
        for label in present_labels
    ]

    metrics["present_labels"] = present_labels

    metrics["macro_f1_present_classes"] = (
        float(np.mean(present_f1_values))
        if present_f1_values
        else 0.0
    )

    normal_index = DEFAULT_ONTOLOGY.labels.index("N")

    abnormal_mask = target_array != normal_index

    abnormal_support = int(
        np.sum(abnormal_mask)
    )

    abnormal_to_normal = int(
        np.sum(
            abnormal_mask
            & (prediction_array == normal_index)
        )
    )

    metrics["abnormal_support"] = abnormal_support

    metrics["abnormal_to_normal_count"] = (
        abnormal_to_normal
    )

    metrics["abnormal_to_normal_fraction"] = (
        abnormal_to_normal / abnormal_support
        if abnormal_support > 0
        else 0.0
    )

    return metrics


def train_fold(
    train_beats: np.ndarray,
    train_labels: np.ndarray,
    validation_beats: np.ndarray,
    validation_labels: np.ndarray,
    output_dir: Path,
    config: TrainingConfig,
) -> dict[str, Any]:
    set_seed(config.seed)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_weights = calculate_class_weights(
        train_labels,
        len(DEFAULT_ONTOLOGY.labels),
        method=config.class_weight_method,
        effective_number_beta=config.effective_number_beta,
    )

    generator = torch.Generator().manual_seed(
        config.seed
    )

    train_loader = DataLoader(
        BeatDataset(
            train_beats,
            train_labels,
        ),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )

    validation_loader = DataLoader(
        BeatDataset(
            validation_beats,
            validation_labels,
        ),
        batch_size=config.batch_size,
        shuffle=False,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = create_detailed_model(
        len(DEFAULT_ONTOLOGY.labels)
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(
            class_weights
        ).to(device)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_macro_f1 = -1.0

    patience = 0

    history: list[dict[str, float]] = []

    for epoch in range(
        1,
        config.epochs + 1,
    ):
        model.train()

        losses: list[float] = []

        for beats, labels in train_loader:
            optimizer.zero_grad()

            logits = model(
                beats.to(device)
            )

            loss = criterion(
                logits,
                labels.to(device),
            )

            loss.backward()
            optimizer.step()

            losses.append(
                float(loss.item())
            )

        validation_metrics = evaluate_fold(
            model,
            validation_loader,
            device,
        )

        history.append(
            {
                "epoch": float(epoch),
                "training_loss": float(
                    np.mean(losses)
                ),
                "validation_macro_f1": float(
                    validation_metrics[
                        "macro_f1"
                    ]
                ),
                "validation_macro_f1_present_classes": float(
                    validation_metrics[
                        "macro_f1_present_classes"
                    ]
                ),
            }
        )

        selection_metric = validation_metrics[
            "macro_f1_present_classes"
        ]

        if selection_metric > best_macro_f1:
            best_macro_f1 = (
                selection_metric
            )

            best_state = {
                name: value.detach()
                .cpu()
                .clone()
                for name, value
                in model.state_dict().items()
            }

            patience = 0

        else:
            patience += 1

        if (
            patience
            >= config.early_stopping_patience
        ):
            break

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a checkpoint"
        )

    model.load_state_dict(
        best_state
    )

    checkpoint_path = (
        output_dir
        / "detailed_classifier.pt"
    )

    torch.save(
        best_state,
        checkpoint_path,
    )

    final_validation_metrics = evaluate_fold(
        model,
        validation_loader,
        device,
    )

    result = {
        "configuration": config.to_dict(),
        "training_seed": config.seed,
        "augmentation_enabled": False,
        "class_weights": (
            class_weights.tolist()
        ),
        "history": history,
        "validation": final_validation_metrics,
    }

    result_path = (
        output_dir
        / "results.json"
    )

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    return result


def summarise(
    values: list[float],
) -> dict[str, Any]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "values": values,
        "mean": float(
            np.mean(array)
        ),
        "std": float(
            np.std(array)
        ),
        "min": float(
            np.min(array)
        ),
        "max": float(
            np.max(array)
        ),
    }

def summarise_supported(
    values: list[float],
    supports: list[int],
) -> dict[str, Any]:
    supported = [
        (value, support)
        for value, support in zip(
            values,
            supports,
            strict=True,
        )
        if support > 0
    ]

    if not supported:
        return {
            "values": [],
            "supports": [],
            "number_of_supported_folds": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }

    supported_values = [
        float(value)
        for value, _ in supported
    ]

    supported_counts = [
        int(support)
        for _, support in supported
    ]

    array = np.asarray(
        supported_values,
        dtype=np.float64,
    )

    return {
        "values": supported_values,
        "supports": supported_counts,
        "number_of_supported_folds": len(
            supported_values
        ),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--records-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--cv-dir",
        type=Path,
        default=(
            Path("research")
            / "detailed_arrhythmia"
            / "outputs"
            / "grouped_patient_cv"
        ),
    )

    arguments = parser.parse_args()

    records_dir = (
        arguments.records_dir
        .expanduser()
        .resolve()
    )

    cv_dir = (
        arguments.cv_dir
        .resolve()
    )

    manifest_path = (
        cv_dir
        / "grouped_cv_manifest.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Grouped CV manifest not found: "
            f"{manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    config = TrainingConfig(
        seed=TRAINING_SEED,
    )

    fold_results: dict[str, Any] = {}

    print()
    print("=" * 70)
    print("Experiment D")
    print("Grouped patient cross validation")
    print("=" * 70)

    print(
        f"Training seed fixed at "
        f"{TRAINING_SEED}"
    )

    print(
        f"Number of folds: "
        f"{len(manifest['folds'])}"
    )

    print(
        "Locked test cohort will NOT "
        "be evaluated."
    )

    for fold in manifest["folds"]:
        fold_number = int(
            fold["fold"]
        )

        train_records = tuple(
            fold["training_records"]
        )

        validation_records = tuple(
            fold["validation_records"]
        )

        print()
        print("=" * 70)
        print(
            f"Fold {fold_number}"
        )
        print("=" * 70)

        print(
            f"Training records: "
            f"{len(train_records)}"
        )

        print(
            f"Validation records: "
            f"{len(validation_records)}"
        )

        print(
            "Validation patients: "
            + ", ".join(
                fold[
                    "validation_patients"
                ]
            )
        )

        train_beats, train_labels, _ = (
            extract_expert_annotated_beats(
                records_dir,
                train_records,
                DEFAULT_ONTOLOGY,
            )
        )

        (
            validation_beats,
            validation_labels,
            _,
        ) = extract_expert_annotated_beats(
            records_dir,
            validation_records,
            DEFAULT_ONTOLOGY,
        )

        print(
            f"Training beats: "
            f"{train_labels.size}"
        )

        print(
            f"Validation beats: "
            f"{validation_labels.size}"
        )

        fold_output = (
            cv_dir
            / f"fold_{fold_number}"
        )

        result_path = (
            fold_output
            / "results.json"
        )

        if result_path.exists():
            print(
                "Existing result found. "
                "Reusing fold."
            )

            result = json.loads(
                result_path.read_text(
                    encoding="utf-8"
                )
            )

        else:
            result = train_fold(
                train_beats,
                train_labels,
                validation_beats,
                validation_labels,
                fold_output,
                config,
            )

        fold_results[
            str(fold_number)
        ] = result

        metrics = result["validation"]

        print(
            f"Fold {fold_number} | "
            f"Macro F1 "
            f"{metrics['macro_f1']:.4f} | "
            f"Present class Macro F1 "
            f"{metrics['macro_f1_present_classes']:.4f} | "
            f"Abnormal to N "
            f"{metrics['abnormal_to_normal_fraction']:.2%}"
        )

    present_macro_f1 = [
        float(
            fold_results[str(index)][
                "validation"
            ][
                "macro_f1_present_classes"
            ]
        )
        for index in range(
            1,
            len(manifest["folds"]) + 1,
        )
    ]

    raw_macro_f1 = [
        float(
            fold_results[str(index)][
                "validation"
            ]["macro_f1"]
        )
        for index in range(
            1,
            len(manifest["folds"]) + 1,
        )
    ]

    abnormal_to_normal = [
        float(
            fold_results[str(index)][
                "validation"
            ][
                "abnormal_to_normal_fraction"
            ]
        )
        for index in range(
            1,
            len(manifest["folds"]) + 1,
        )
    ]

    per_class: dict[str, Any] = {}

    for label in DEFAULT_ONTOLOGY.labels:
        f1_values = [
            float(
                fold_results[str(index)]["validation"][
                    "per_class"
                ][label]["f1"]
            )
            for index in range(
                1,
                len(manifest["folds"]) + 1,
            )
        ]

    recall_values = [
        float(
            fold_results[str(index)]["validation"][
                "per_class"
            ][label]["recall"]
        )
        for index in range(
            1,
            len(manifest["folds"]) + 1,
        )
    ]

    support_values = [
        int(
            fold_results[str(index)]["validation"][
                "per_class"
            ][label]["support"]
        )
        for index in range(
            1,
            len(manifest["folds"]) + 1,
        )
    ]

    per_class[label] = {
        "f1_all_folds": summarise(
            f1_values
        ),
        "f1_support_aware": summarise_supported(
            f1_values,
            support_values,
        ),
        "recall_all_folds": summarise(
            recall_values
        ),
        "recall_support_aware": summarise_supported(
            recall_values,
            support_values,
        ),
        "support_by_fold": support_values,
    }

    summary = {
        "experiment": (
            "grouped_patient_cross_validation"
        ),
        "training_seed": TRAINING_SEED,
        "n_folds": len(
            manifest["folds"]
        ),
        "locked_test_evaluated": False,
        "macro_f1": summarise(
            raw_macro_f1
        ),
        "macro_f1_present_classes": summarise(
            present_macro_f1
        ),
        "abnormal_to_normal_fraction": summarise(
            abnormal_to_normal
        ),
        "per_class": per_class,
        "fold_results": {
            fold: {
                "validation": result[
                    "validation"
                ],
                "validation_patients": (
                    manifest["folds"][
                        int(fold) - 1
                    ][
                        "validation_patients"
                    ]
                ),
            }
            for fold, result
            in fold_results.items()
        },
    }

    summary_path = (
        cv_dir
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print(
        "Grouped patient CV complete"
    )
    print("=" * 70)

    print(
        "Present class Macro F1: "
        f"{summary['macro_f1_present_classes']['mean']:.4f} "
        "± "
        f"{summary['macro_f1_present_classes']['std']:.4f}"
    )

    print(
        "Abnormal to N: "
        f"{summary['abnormal_to_normal_fraction']['mean']:.2%} "
        "± "
        f"{summary['abnormal_to_normal_fraction']['std']:.2%}"
    )

    print()
    print(
        f"Summary written to: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()