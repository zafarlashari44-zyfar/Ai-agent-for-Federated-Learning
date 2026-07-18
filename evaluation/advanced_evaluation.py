from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from fl_ecg_orchestrator.data.loader import (
    CLASS_NAMES,
    load_config,
    load_global_test_data,
    resolve_project_path,
)
from fl_ecg_orchestrator.model.cnn1d import create_model


CLASS_IDS = sorted(CLASS_NAMES)
CLASS_LABELS = [CLASS_NAMES[class_id] for class_id in CLASS_IDS]


def load_checkpoint_model(
    checkpoint_path: str | Path,
    config_path: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    config = load_config(config_path)
    input_length = int(config["data"]["input_length"])
    num_classes = int(config["data"]["num_classes"])

    model = create_model(
        input_length=input_length,
        num_classes=num_classes,
    ).to(device)

    checkpoint_path = resolve_project_path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, checkpoint


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_labels: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(features)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        all_labels.append(labels.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())

    return (
        np.concatenate(all_labels).astype(np.int64),
        np.concatenate(all_predictions).astype(np.int64),
        np.concatenate(all_probabilities).astype(np.float64),
    )


def safe_multiclass_auc(
    y_true: np.ndarray,
    y_probability: np.ndarray,
) -> tuple[dict[str, float | None], float | None, float | None]:
    y_binary = label_binarize(y_true, classes=CLASS_IDS)
    per_class: dict[str, float | None] = {}

    valid_curves: list[tuple[np.ndarray, np.ndarray]] = []

    for class_position, class_id in enumerate(CLASS_IDS):
        target = y_binary[:, class_position]

        if np.unique(target).size < 2:
            per_class[CLASS_NAMES[class_id]] = None
            continue

        fpr, tpr, _ = roc_curve(
            target,
            y_probability[:, class_position],
        )
        class_auc = float(auc(fpr, tpr))
        per_class[CLASS_NAMES[class_id]] = class_auc
        valid_curves.append((fpr, tpr))

    micro_auc: float | None = None
    if np.unique(y_binary.ravel()).size == 2:
        micro_fpr, micro_tpr, _ = roc_curve(
            y_binary.ravel(),
            y_probability.ravel(),
        )
        micro_auc = float(auc(micro_fpr, micro_tpr))

    macro_auc: float | None = None
    if valid_curves:
        all_fpr = np.unique(
            np.concatenate([curve[0] for curve in valid_curves])
        )
        mean_tpr = np.zeros_like(all_fpr)

        for fpr, tpr in valid_curves:
            mean_tpr += np.interp(all_fpr, fpr, tpr)

        mean_tpr /= len(valid_curves)
        macro_auc = float(auc(all_fpr, mean_tpr))

    return per_class, micro_auc, macro_auc


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, Any]:
    precision_macro = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    recall_macro = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    precision_weighted = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    recall_weighted = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    per_precision, per_recall, per_f1, per_support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=CLASS_IDS,
            zero_division=0,
        )
    )

    per_class_auc, micro_auc, macro_auc = safe_multiclass_auc(
        y_true,
        y_probability,
    )

    per_class = {}

    for index, class_id in enumerate(CLASS_IDS):
        class_mask = y_true == class_id
        class_accuracy = (
            float(np.mean(y_pred[class_mask] == class_id))
            if np.any(class_mask)
            else None
        )

        per_class[CLASS_NAMES[class_id]] = {
            "class_id": int(class_id),
            "precision": float(per_precision[index]),
            "recall": float(per_recall[index]),
            "f1_score": float(per_f1[index]),
            "support": int(per_support[index]),
            "class_accuracy": class_accuracy,
            "roc_auc": per_class_auc[CLASS_NAMES[class_id]],
        }

    metrics: dict[str, Any] = {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
        "matthews_correlation_coefficient": float(
            matthews_corrcoef(y_true, y_pred)
        ),
        "log_loss": float(
            log_loss(
                y_true,
                y_probability,
                labels=CLASS_IDS,
            )
        ),
        "micro_roc_auc": micro_auc,
        "macro_roc_auc": macro_auc,
        "per_class": per_class,
    }

    return metrics


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    output_path: Path,
    normalized: bool,
) -> None:
    display_matrix = matrix.astype(np.float64)

    if normalized:
        row_sums = display_matrix.sum(axis=1, keepdims=True)
        display_matrix = np.divide(
            display_matrix,
            row_sums,
            out=np.zeros_like(display_matrix),
            where=row_sums != 0,
        )

    plt.figure(figsize=(8, 7))
    image = plt.imshow(display_matrix, interpolation="nearest")
    plt.colorbar(image)
    plt.xticks(range(len(CLASS_LABELS)), CLASS_LABELS)
    plt.yticks(range(len(CLASS_LABELS)), CLASS_LABELS)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(
        "Normalized Confusion Matrix"
        if normalized
        else "Confusion Matrix"
    )

    threshold = (
        float(display_matrix.max()) / 2
        if display_matrix.size
        else 0.0
    )

    for row in range(display_matrix.shape[0]):
        for column in range(display_matrix.shape[1]):
            value = display_matrix[row, column]
            text = f"{value:.2f}" if normalized else f"{int(value)}"
            plt.text(
                column,
                row,
                text,
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if value > threshold else "black",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close()


def save_roc_plot(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    output_path: Path,
) -> dict[str, float | None]:
    y_binary = label_binarize(y_true, classes=CLASS_IDS)
    auc_scores: dict[str, float | None] = {}

    plt.figure(figsize=(9, 8))

    valid_curves: list[tuple[np.ndarray, np.ndarray]] = []

    for class_position, class_id in enumerate(CLASS_IDS):
        target = y_binary[:, class_position]
        class_name = CLASS_NAMES[class_id]

        if np.unique(target).size < 2:
            auc_scores[class_name] = None
            continue

        fpr, tpr, _ = roc_curve(
            target,
            y_probability[:, class_position],
        )
        score = float(auc(fpr, tpr))
        auc_scores[class_name] = score
        valid_curves.append((fpr, tpr))

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{class_name} (AUC={score:.3f})",
        )

    if np.unique(y_binary.ravel()).size == 2:
        micro_fpr, micro_tpr, _ = roc_curve(
            y_binary.ravel(),
            y_probability.ravel(),
        )
        micro_score = float(auc(micro_fpr, micro_tpr))
        auc_scores["micro_average"] = micro_score

        plt.plot(
            micro_fpr,
            micro_tpr,
            linestyle="--",
            linewidth=2.5,
            label=f"Micro-average (AUC={micro_score:.3f})",
        )
    else:
        auc_scores["micro_average"] = None

    if valid_curves:
        all_fpr = np.unique(
            np.concatenate([curve[0] for curve in valid_curves])
        )
        mean_tpr = np.zeros_like(all_fpr)

        for fpr, tpr in valid_curves:
            mean_tpr += np.interp(all_fpr, fpr, tpr)

        mean_tpr /= len(valid_curves)
        macro_score = float(auc(all_fpr, mean_tpr))
        auc_scores["macro_average"] = macro_score

        plt.plot(
            all_fpr,
            mean_tpr,
            linestyle=":",
            linewidth=3,
            label=f"Macro-average (AUC={macro_score:.3f})",
        )
    else:
        auc_scores["macro_average"] = None

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curves — One-vs-Rest")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close()

    return auc_scores


def export_evaluation(
    checkpoint_path: str | Path,
    config_path: str | Path = "fl_ecg_orchestrator/config/config.yaml",
    output_dir: str | Path = "outputs/evaluation",
    device_name: str | None = None,
) -> Path:
    device = torch.device(
        device_name
        if device_name
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    output_path = resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_checkpoint_model(
        checkpoint_path,
        config_path,
        device,
    )

    global_test = load_global_test_data(config_path)
    y_true, y_pred, y_probability = collect_predictions(
        model,
        global_test["loader"],
        device,
    )

    metrics = calculate_metrics(
        y_true,
        y_pred,
        y_probability,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_IDS,
    )

    save_confusion_matrix_plot(
        matrix,
        output_path / "confusion_matrix.png",
        normalized=False,
    )
    save_confusion_matrix_plot(
        matrix,
        output_path / "confusion_matrix_normalized.png",
        normalized=True,
    )

    auc_scores = save_roc_plot(
        y_true,
        y_probability,
        output_path / "roc_curve_multiclass.png",
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        target_names=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        target_names=CLASS_LABELS,
        zero_division=0,
    )

    with (output_path / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    with (output_path / "classification_report.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report_dict, file, indent=2)

    with (output_path / "classification_report.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["class", "precision", "recall", "f1_score", "support"]
        )

        for label, values in report_dict.items():
            if not isinstance(values, dict):
                continue
            writer.writerow(
                [
                    label,
                    values.get("precision"),
                    values.get("recall"),
                    values.get("f1-score"),
                    values.get("support"),
                ]
            )

    with (output_path / "per_class_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "class",
                "class_id",
                "precision",
                "recall",
                "f1_score",
                "support",
                "class_accuracy",
                "roc_auc",
            ]
        )

        for class_name, values in metrics["per_class"].items():
            writer.writerow(
                [
                    class_name,
                    values["class_id"],
                    values["precision"],
                    values["recall"],
                    values["f1_score"],
                    values["support"],
                    values["class_accuracy"],
                    values["roc_auc"],
                ]
            )

    with (output_path / "auc_scores.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["curve", "auc"])
        for label, score in auc_scores.items():
            writer.writerow([label, score])

    np.savez_compressed(
        output_path / "evaluation_predictions.npz",
        true_labels=y_true,
        predictions=y_pred,
        probabilities=y_probability,
        confusion_matrix=matrix,
    )

    checkpoint_round = checkpoint.get("round")
    checkpoint_metrics = checkpoint.get("metrics", {})

    summary_lines = [
        "Advanced ECG Model Evaluation",
        "=" * 31,
        f"Checkpoint: {resolve_project_path(checkpoint_path)}",
        f"Device: {device}",
        f"Samples: {metrics['samples']}",
        f"Checkpoint round: {checkpoint_round}",
        "",
        "Overall metrics",
        "-" * 15,
        f"Accuracy: {metrics['accuracy']:.6f}",
        f"Balanced accuracy: {metrics['balanced_accuracy']:.6f}",
        f"Macro precision: {metrics['macro_precision']:.6f}",
        f"Macro recall: {metrics['macro_recall']:.6f}",
        f"Macro F1: {metrics['macro_f1']:.6f}",
        f"Weighted F1: {metrics['weighted_f1']:.6f}",
        f"MCC: {metrics['matthews_correlation_coefficient']:.6f}",
        f"Log loss: {metrics['log_loss']:.6f}",
        f"Micro ROC AUC: {metrics['micro_roc_auc']}",
        f"Macro ROC AUC: {metrics['macro_roc_auc']}",
        "",
        "Classification report",
        "-" * 21,
        report_text,
        "",
        "Checkpoint metrics",
        "-" * 18,
        json.dumps(checkpoint_metrics, indent=2),
    ]

    (output_path / "evaluation_summary.txt").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    return output_path
