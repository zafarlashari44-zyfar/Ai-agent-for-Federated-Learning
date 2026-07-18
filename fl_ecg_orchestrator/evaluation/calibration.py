from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fl_ecg_orchestrator.evaluation.global_evaluator import GlobalEvaluator


class CalibrationEvaluator:
    """Evaluate confidence calibration for a trained multiclass ECG model."""

    def __init__(
        self,
        config_path: str = "fl_ecg_orchestrator/config/config.yaml",
        device: str | None = None,
        num_bins: int = 15,
    ) -> None:
        if num_bins < 2:
            raise ValueError("num_bins must be at least 2.")

        self.global_evaluator = GlobalEvaluator(
            config_path=config_path,
            device=device,
        )
        self.num_bins = int(num_bins)

    def run(
        self,
        checkpoint_path: str | Path,
        output_dir: str | Path = "outputs/calibration",
    ) -> dict[str, Any]:
        checkpoint_path = Path(checkpoint_path).resolve()
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        checkpoint = self.global_evaluator.load_checkpoint(checkpoint_path)
        true_labels, predictions, probabilities = (
            self.global_evaluator.collect_predictions()
        )

        confidences = probabilities.max(axis=1)
        correctness = (predictions == true_labels).astype(np.float64)

        bin_data = self._calculate_bins(
            confidences=confidences,
            correctness=correctness,
        )

        expected_calibration_error = float(
            sum(
                row["proportion"]
                * abs(row["accuracy"] - row["confidence"])
                for row in bin_data
            )
        )

        maximum_calibration_error = float(
            max(
                (
                    abs(row["accuracy"] - row["confidence"])
                    for row in bin_data
                    if row["count"] > 0
                ),
                default=0.0,
            )
        )

        one_hot_targets = np.eye(
            probabilities.shape[1],
            dtype=np.float64,
        )[true_labels]

        multiclass_brier_score = float(
            np.mean(
                np.sum(
                    (probabilities - one_hot_targets) ** 2,
                    axis=1,
                )
            )
        )

        accuracy = float(correctness.mean())
        average_confidence = float(confidences.mean())
        confidence_gap = float(average_confidence - accuracy)

        metrics: dict[str, Any] = {
            "samples": int(len(true_labels)),
            "num_bins": self.num_bins,
            "accuracy": accuracy,
            "average_confidence": average_confidence,
            "confidence_gap": confidence_gap,
            "expected_calibration_error": expected_calibration_error,
            "maximum_calibration_error": maximum_calibration_error,
            "multiclass_brier_score": multiclass_brier_score,
            "checkpoint": str(checkpoint_path),
            "round": checkpoint.get("round"),
        }

        self._save_reliability_diagram(
            bin_data=bin_data,
            output_path=output_path / "reliability_diagram.png",
        )
        self._save_confidence_histogram(
            confidences=confidences,
            correctness=correctness,
            output_path=output_path / "confidence_histogram.png",
        )
        self._save_confidence_vs_accuracy(
            bin_data=bin_data,
            output_path=output_path / "confidence_vs_accuracy.png",
        )
        self._save_bins_csv(
            bin_data=bin_data,
            output_path=output_path / "calibration_bins.csv",
        )
        self._save_json(
            payload=metrics,
            output_path=output_path / "calibration_metrics.json",
        )
        self._save_predictions(
            true_labels=true_labels,
            predictions=predictions,
            probabilities=probabilities,
            confidences=confidences,
            correctness=correctness,
            output_path=output_path / "calibration_predictions.npz",
        )
        self._save_summary(
            metrics=metrics,
            output_path=output_path / "calibration_summary.txt",
        )

        return {
            "metrics": metrics,
            "output_dir": str(output_path),
        }

    def _calculate_bins(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray,
    ) -> list[dict[str, Any]]:
        boundaries = np.linspace(
            0.0,
            1.0,
            self.num_bins + 1,
        )

        rows: list[dict[str, Any]] = []
        total = max(len(confidences), 1)

        for index in range(self.num_bins):
            lower = float(boundaries[index])
            upper = float(boundaries[index + 1])

            if index == self.num_bins - 1:
                mask = (confidences >= lower) & (confidences <= upper)
            else:
                mask = (confidences >= lower) & (confidences < upper)

            count = int(mask.sum())

            if count > 0:
                accuracy = float(correctness[mask].mean())
                confidence = float(confidences[mask].mean())
            else:
                accuracy = 0.0
                confidence = float((lower + upper) / 2.0)

            rows.append(
                {
                    "bin": index + 1,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "count": count,
                    "proportion": float(count / total),
                    "accuracy": accuracy,
                    "confidence": confidence,
                    "absolute_gap": float(abs(accuracy - confidence)),
                }
            )

        return rows

    def _save_reliability_diagram(
        self,
        bin_data: list[dict[str, Any]],
        output_path: Path,
    ) -> None:
        confidences = np.asarray(
            [row["confidence"] for row in bin_data],
            dtype=np.float64,
        )
        accuracies = np.asarray(
            [row["accuracy"] for row in bin_data],
            dtype=np.float64,
        )

        figure, axis = plt.subplots(figsize=(8, 7))
        axis.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
        axis.plot(confidences, accuracies, marker="o", label="Model")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Mean predicted confidence")
        axis.set_ylabel("Observed accuracy")
        axis.set_title("Reliability Diagram")
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi=250, bbox_inches="tight")
        plt.close(figure)

    def _save_confidence_histogram(
        self,
        confidences: np.ndarray,
        correctness: np.ndarray,
        output_path: Path,
    ) -> None:
        figure, axis = plt.subplots(figsize=(9, 6))

        axis.hist(
            confidences[correctness == 1],
            bins=20,
            alpha=0.65,
            label="Correct predictions",
        )
        axis.hist(
            confidences[correctness == 0],
            bins=20,
            alpha=0.65,
            label="Incorrect predictions",
        )

        axis.set_xlim(0, 1)
        axis.set_xlabel("Prediction confidence")
        axis.set_ylabel("Number of samples")
        axis.set_title("Confidence Distribution")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi=250, bbox_inches="tight")
        plt.close(figure)

    def _save_confidence_vs_accuracy(
        self,
        bin_data: list[dict[str, Any]],
        output_path: Path,
    ) -> None:
        labels = [
            f"{row['lower_bound']:.2f}-{row['upper_bound']:.2f}"
            for row in bin_data
        ]
        positions = np.arange(len(bin_data))
        width = 0.38

        confidence_values = [row["confidence"] for row in bin_data]
        accuracy_values = [row["accuracy"] for row in bin_data]

        figure, axis = plt.subplots(figsize=(12, 6))
        axis.bar(
            positions - width / 2,
            confidence_values,
            width,
            label="Mean confidence",
        )
        axis.bar(
            positions + width / 2,
            accuracy_values,
            width,
            label="Observed accuracy",
        )

        axis.set_ylim(0, 1)
        axis.set_xlabel("Confidence bin")
        axis.set_ylabel("Score")
        axis.set_title("Confidence vs Accuracy by Bin")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=45, ha="right")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi=250, bbox_inches="tight")
        plt.close(figure)

    def _save_bins_csv(
        self,
        bin_data: list[dict[str, Any]],
        output_path: Path,
    ) -> None:
        fieldnames = [
            "bin",
            "lower_bound",
            "upper_bound",
            "count",
            "proportion",
            "accuracy",
            "confidence",
            "absolute_gap",
        ]

        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(bin_data)

    def _save_json(
        self,
        payload: dict[str, Any],
        output_path: Path,
    ) -> None:
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    def _save_predictions(
        self,
        true_labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        confidences: np.ndarray,
        correctness: np.ndarray,
        output_path: Path,
    ) -> None:
        np.savez_compressed(
            output_path,
            true_labels=true_labels,
            predictions=predictions,
            probabilities=probabilities,
            confidences=confidences,
            correctness=correctness,
        )

    def _save_summary(
        self,
        metrics: dict[str, Any],
        output_path: Path,
    ) -> None:
        confidence_status = (
            "overconfident"
            if metrics["confidence_gap"] > 0
            else "underconfident"
            if metrics["confidence_gap"] < 0
            else "perfectly aligned"
        )

        summary = "\n".join(
            [
                "ECG Model Calibration Evaluation",
                "=" * 32,
                "",
                f"Checkpoint: {metrics['checkpoint']}",
                f"Round: {metrics['round']}",
                f"Samples: {metrics['samples']}",
                f"Bins: {metrics['num_bins']}",
                "",
                f"Accuracy: {metrics['accuracy']:.6f}",
                f"Average confidence: {metrics['average_confidence']:.6f}",
                f"Confidence gap: {metrics['confidence_gap']:.6f}",
                (
                    "Expected Calibration Error (ECE): "
                    f"{metrics['expected_calibration_error']:.6f}"
                ),
                (
                    "Maximum Calibration Error (MCE): "
                    f"{metrics['maximum_calibration_error']:.6f}"
                ),
                (
                    "Multiclass Brier score: "
                    f"{metrics['multiclass_brier_score']:.6f}"
                ),
                "",
                f"Calibration tendency: {confidence_status}",
            ]
        )

        output_path.write_text(summary, encoding="utf-8")
