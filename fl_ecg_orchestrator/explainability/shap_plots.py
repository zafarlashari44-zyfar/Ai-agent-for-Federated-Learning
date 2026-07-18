from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from fl_ecg_orchestrator.data.loader import CLASS_NAMES, resolve_project_path


def feature_names(input_length: int) -> list[str]:
    return [f"ECG sample {index:03d}" for index in range(input_length)]


def save_global_feature_importance(
    results: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    output_path = resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    importance = np.abs(
        results["predicted_shap_values"]
    ).mean(axis=0)

    order = np.argsort(importance)[::-1]
    csv_path = output_path / "global_feature_importance.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["rank", "feature_index", "feature_name", "mean_abs_shap"])

        for rank, feature_index in enumerate(order, start=1):
            writer.writerow(
                [
                    rank,
                    int(feature_index),
                    f"ECG sample {int(feature_index):03d}",
                    float(importance[feature_index]),
                ]
            )

    top_count = min(30, len(order))
    top_indices = order[:top_count][::-1]

    plt.figure(figsize=(10, 8))
    plt.barh(
        [f"Sample {index}" for index in top_indices],
        importance[top_indices],
    )
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("ECG sample index")
    plt.title("Global ECG Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path / "global_feature_importance.png", dpi=220)
    plt.close()

    return csv_path


def save_summary_plot(
    results: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    output_path = resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    path = output_path / "shap_summary.png"

    shap.summary_plot(
        results["predicted_shap_values"],
        results["features"],
        feature_names=feature_names(results["features"].shape[1]),
        show=False,
        max_display=25,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()

    return path


def save_class_importance_plots(
    results: dict[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    output_path = resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    for class_id, class_name in CLASS_NAMES.items():
        mask = results["predictions"] == class_id
        if not np.any(mask):
            continue

        importance = np.abs(
            results["shap_values"][mask, class_id, :]
        ).mean(axis=0)

        order = np.argsort(importance)[::-1][:20][::-1]
        path = output_path / f"class_{class_name}_importance.png"

        plt.figure(figsize=(10, 7))
        plt.barh(
            [f"Sample {index}" for index in order],
            importance[order],
        )
        plt.xlabel("Mean absolute SHAP value")
        plt.ylabel("ECG sample index")
        plt.title(f"SHAP Feature Importance — Class {class_name}")
        plt.tight_layout()
        plt.savefig(path, dpi=220)
        plt.close()

        saved_paths.append(path)

    return saved_paths


def save_individual_explanation(
    results: dict[str, Any],
    sample_position: int,
    output_dir: str | Path,
    top_features: int = 20,
) -> Path:
    if not 0 <= sample_position < len(results["features"]):
        raise IndexError(
            f"sample_position must be between 0 and "
            f"{len(results['features']) - 1}"
        )

    output_path = resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prediction = int(results["predictions"][sample_position])
    true_label = int(results["true_labels"][sample_position])
    probability = float(
        results["probabilities"][sample_position, prediction]
    )
    values = results["shap_values"][sample_position, prediction]
    feature_values = results["features"][sample_position]

    order = np.argsort(np.abs(values))[::-1][:top_features][::-1]
    path = output_path / (
        f"sample_{sample_position:04d}_"
        f"true_{CLASS_NAMES[true_label]}_"
        f"pred_{CLASS_NAMES[prediction]}.png"
    )

    labels = [
        f"{index}: {feature_values[index]:.3f}"
        for index in order
    ]

    plt.figure(figsize=(11, 8))
    plt.barh(labels, values[order])
    plt.axvline(0.0, linewidth=1)
    plt.xlabel("SHAP contribution to predicted probability")
    plt.ylabel("ECG sample index and value")
    plt.title(
        f"Individual SHAP Explanation | "
        f"True={CLASS_NAMES[true_label]} | "
        f"Predicted={CLASS_NAMES[prediction]} | "
        f"Confidence={probability:.2%}"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()

    return path
