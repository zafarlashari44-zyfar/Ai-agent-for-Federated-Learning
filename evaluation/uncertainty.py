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
from torch import nn

from fl_ecg_orchestrator.evaluation.global_evaluator import GlobalEvaluator


CLASS_NAMES = {
    0: "N",
    1: "S",
    2: "V",
    3: "F",
    4: "Q",
}


class UncertaintyEvaluator:
    """Monte Carlo Dropout uncertainty evaluation for ECG classification."""

    def __init__(
        self,
        config_path: str = "fl_ecg_orchestrator/config/config.yaml",
        device: str | None = None,
        mc_samples: int = 30,
    ) -> None:
        if mc_samples < 2:
            raise ValueError("mc_samples must be at least 2.")

        self.global_evaluator = GlobalEvaluator(
            config_path=config_path,
            device=device,
        )
        self.mc_samples = int(mc_samples)

    def run(
        self,
        checkpoint_path: str | Path,
        output_dir: str | Path = "outputs/uncertainty",
    ) -> dict[str, Any]:
        checkpoint_path = Path(checkpoint_path).resolve()
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        checkpoint = self.global_evaluator.load_checkpoint(checkpoint_path)

        dropout_layers = self._count_dropout_layers(
            self.global_evaluator.model
        )

        if dropout_layers == 0:
            raise RuntimeError(
                "The model contains no Dropout layers, so Monte Carlo "
                "Dropout cannot estimate uncertainty. Add nn.Dropout to "
                "the CNN model before using this module."
            )

        results = self._collect_mc_predictions()

        true_labels = results["true_labels"]
        predictions = results["predictions"]
        mean_probabilities = results["mean_probabilities"]
        predictive_entropy = results["predictive_entropy"]
        mutual_information = results["mutual_information"]
        predictive_variance = results["predictive_variance"]
        confidence = results["confidence"]

        correctness = (predictions == true_labels).astype(np.int64)
        normalized_entropy = predictive_entropy / np.log(
            mean_probabilities.shape[1]
        )

        uncertainty_level = np.where(
            normalized_entropy >= 0.60,
            "High",
            np.where(
                normalized_entropy >= 0.30,
                "Medium",
                "Low",
            ),
        )

        recommendation = np.where(
            uncertainty_level == "High",
            "Manual cardiologist review recommended.",
            np.where(
                uncertainty_level == "Medium",
                "Review alongside clinical context.",
                "Prediction suitable for routine review.",
            ),
        )

        metrics = self._calculate_metrics(
            true_labels=true_labels,
            predictions=predictions,
            confidence=confidence,
            predictive_entropy=predictive_entropy,
            normalized_entropy=normalized_entropy,
            mutual_information=mutual_information,
            predictive_variance=predictive_variance,
            uncertainty_level=uncertainty_level,
            dropout_layers=dropout_layers,
            checkpoint=checkpoint,
            checkpoint_path=checkpoint_path,
        )

        self._save_predictions_csv(
            true_labels=true_labels,
            predictions=predictions,
            confidence=confidence,
            predictive_entropy=predictive_entropy,
            normalized_entropy=normalized_entropy,
            mutual_information=mutual_information,
            predictive_variance=predictive_variance,
            uncertainty_level=uncertainty_level,
            recommendation=recommendation,
            correctness=correctness,
            output_path=output_path / "uncertainty_predictions.csv",
        )

        self._save_entropy_histogram(
            normalized_entropy=normalized_entropy,
            correctness=correctness,
            output_path=output_path / "entropy_histogram.png",
        )

        self._save_uncertainty_distribution(
            uncertainty_level=uncertainty_level,
            output_path=output_path / "uncertainty_distribution.png",
        )

        self._save_confidence_vs_uncertainty(
            confidence=confidence,
            normalized_entropy=normalized_entropy,
            correctness=correctness,
            output_path=output_path / "confidence_vs_uncertainty.png",
        )

        self._save_json(
            payload=metrics,
            output_path=output_path / "uncertainty_metrics.json",
        )

        self._save_summary(
            metrics=metrics,
            output_path=output_path / "uncertainty_summary.txt",
        )

        np.savez_compressed(
            output_path / "uncertainty_predictions.npz",
            true_labels=true_labels,
            predictions=predictions,
            mean_probabilities=mean_probabilities,
            confidence=confidence,
            predictive_entropy=predictive_entropy,
            normalized_entropy=normalized_entropy,
            mutual_information=mutual_information,
            predictive_variance=predictive_variance,
            correctness=correctness,
        )

        return {
            "metrics": metrics,
            "output_dir": str(output_path),
        }

    @torch.no_grad()
    def _collect_mc_predictions(self) -> dict[str, np.ndarray]:
        model = self.global_evaluator.model
        loader = self.global_evaluator.test_data["loader"]
        device = self.global_evaluator.device

        model.eval()
        self._enable_dropout(model)

        all_labels: list[np.ndarray] = []
        all_mean_probabilities: list[np.ndarray] = []
        all_predictive_entropy: list[np.ndarray] = []
        all_mutual_information: list[np.ndarray] = []
        all_predictive_variance: list[np.ndarray] = []

        for batch in loader:
            if len(batch) < 2:
                raise ValueError(
                    "Global test loader must return features and labels."
                )

            features = batch[0].to(
                device,
                dtype=torch.float32,
            )
            labels = batch[1].to(
                device,
                dtype=torch.long,
            )

            probability_samples: list[torch.Tensor] = []

            for _ in range(self.mc_samples):
                logits = model(features)
                probabilities = torch.softmax(
                    logits,
                    dim=1,
                ).to(torch.float64)

                probabilities = probabilities / probabilities.sum(
                    dim=1,
                    keepdim=True,
                ).clamp_min(1e-12)

                probability_samples.append(probabilities)

            stacked = torch.stack(
                probability_samples,
                dim=0,
            )

            mean_probabilities = stacked.mean(dim=0)
            predictive_variance = stacked.var(
                dim=0,
                unbiased=False,
            ).mean(dim=1)

            predictive_entropy = -torch.sum(
                mean_probabilities
                * torch.log(mean_probabilities.clamp_min(1e-12)),
                dim=1,
            )

            sample_entropies = -torch.sum(
                stacked * torch.log(stacked.clamp_min(1e-12)),
                dim=2,
            )

            expected_entropy = sample_entropies.mean(dim=0)
            mutual_information = predictive_entropy - expected_entropy

            all_labels.append(
                labels.detach().cpu().numpy()
            )
            all_mean_probabilities.append(
                mean_probabilities.detach().cpu().numpy()
            )
            all_predictive_entropy.append(
                predictive_entropy.detach().cpu().numpy()
            )
            all_mutual_information.append(
                mutual_information.detach().cpu().numpy()
            )
            all_predictive_variance.append(
                predictive_variance.detach().cpu().numpy()
            )

        model.eval()

        if not all_labels:
            raise RuntimeError(
                "No samples were returned by the global test loader."
            )

        true_labels = np.concatenate(all_labels).astype(np.int64)
        mean_probabilities = np.concatenate(
            all_mean_probabilities
        ).astype(np.float64)

        predictions = mean_probabilities.argmax(axis=1).astype(np.int64)
        confidence = mean_probabilities.max(axis=1).astype(np.float64)

        return {
            "true_labels": true_labels,
            "predictions": predictions,
            "mean_probabilities": mean_probabilities,
            "confidence": confidence,
            "predictive_entropy": np.concatenate(
                all_predictive_entropy
            ).astype(np.float64),
            "mutual_information": np.concatenate(
                all_mutual_information
            ).astype(np.float64),
            "predictive_variance": np.concatenate(
                all_predictive_variance
            ).astype(np.float64),
        }

    def _enable_dropout(self, model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(
                module,
                (
                    nn.Dropout,
                    nn.Dropout1d,
                    nn.Dropout2d,
                    nn.Dropout3d,
                    nn.AlphaDropout,
                    nn.FeatureAlphaDropout,
                ),
            ):
                module.train()

    def _count_dropout_layers(self, model: nn.Module) -> int:
        return sum(
            1
            for module in model.modules()
            if isinstance(
                module,
                (
                    nn.Dropout,
                    nn.Dropout1d,
                    nn.Dropout2d,
                    nn.Dropout3d,
                    nn.AlphaDropout,
                    nn.FeatureAlphaDropout,
                ),
            )
        )

    def _calculate_metrics(
        self,
        true_labels: np.ndarray,
        predictions: np.ndarray,
        confidence: np.ndarray,
        predictive_entropy: np.ndarray,
        normalized_entropy: np.ndarray,
        mutual_information: np.ndarray,
        predictive_variance: np.ndarray,
        uncertainty_level: np.ndarray,
        dropout_layers: int,
        checkpoint: dict[str, Any],
        checkpoint_path: Path,
    ) -> dict[str, Any]:
        correctness = predictions == true_labels

        correlation = float(
            np.corrcoef(confidence, normalized_entropy)[0, 1]
        )

        if np.isnan(correlation):
            correlation = 0.0

        level_counts = {
            level: int(np.sum(uncertainty_level == level))
            for level in ("Low", "Medium", "High")
        }

        level_accuracy: dict[str, float | None] = {}

        for level in ("Low", "Medium", "High"):
            mask = uncertainty_level == level
            level_accuracy[level] = (
                float(np.mean(correctness[mask]))
                if np.any(mask)
                else None
            )

        return {
            "checkpoint": str(checkpoint_path),
            "round": checkpoint.get("round"),
            "samples": int(len(true_labels)),
            "mc_samples": self.mc_samples,
            "dropout_layers": dropout_layers,
            "accuracy": float(np.mean(correctness)),
            "average_confidence": float(np.mean(confidence)),
            "mean_predictive_entropy": float(
                np.mean(predictive_entropy)
            ),
            "mean_normalized_entropy": float(
                np.mean(normalized_entropy)
            ),
            "mean_mutual_information": float(
                np.mean(mutual_information)
            ),
            "mean_predictive_variance": float(
                np.mean(predictive_variance)
            ),
            "confidence_uncertainty_correlation": correlation,
            "uncertainty_counts": level_counts,
            "accuracy_by_uncertainty_level": level_accuracy,
        }

    def _save_predictions_csv(
        self,
        true_labels: np.ndarray,
        predictions: np.ndarray,
        confidence: np.ndarray,
        predictive_entropy: np.ndarray,
        normalized_entropy: np.ndarray,
        mutual_information: np.ndarray,
        predictive_variance: np.ndarray,
        uncertainty_level: np.ndarray,
        recommendation: np.ndarray,
        correctness: np.ndarray,
        output_path: Path,
    ) -> None:
        fieldnames = [
            "sample_index",
            "true_class",
            "predicted_class",
            "confidence",
            "predictive_entropy",
            "normalized_entropy",
            "mutual_information",
            "predictive_variance",
            "uncertainty_level",
            "correct",
            "recommendation",
        ]

        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for index in range(len(true_labels)):
                writer.writerow(
                    {
                        "sample_index": index,
                        "true_class": CLASS_NAMES.get(
                            int(true_labels[index]),
                            str(int(true_labels[index])),
                        ),
                        "predicted_class": CLASS_NAMES.get(
                            int(predictions[index]),
                            str(int(predictions[index])),
                        ),
                        "confidence": float(confidence[index]),
                        "predictive_entropy": float(
                            predictive_entropy[index]
                        ),
                        "normalized_entropy": float(
                            normalized_entropy[index]
                        ),
                        "mutual_information": float(
                            mutual_information[index]
                        ),
                        "predictive_variance": float(
                            predictive_variance[index]
                        ),
                        "uncertainty_level": str(
                            uncertainty_level[index]
                        ),
                        "correct": int(correctness[index]),
                        "recommendation": str(recommendation[index]),
                    }
                )

    def _save_entropy_histogram(
        self,
        normalized_entropy: np.ndarray,
        correctness: np.ndarray,
        output_path: Path,
    ) -> None:
        figure, axis = plt.subplots(figsize=(9, 6))

        axis.hist(
            normalized_entropy[correctness == 1],
            bins=25,
            alpha=0.65,
            label="Correct predictions",
        )
        axis.hist(
            normalized_entropy[correctness == 0],
            bins=25,
            alpha=0.65,
            label="Incorrect predictions",
        )

        axis.set_xlim(0, 1)
        axis.set_xlabel("Normalized predictive entropy")
        axis.set_ylabel("Number of samples")
        axis.set_title("Predictive Entropy Distribution")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi=250, bbox_inches="tight")
        plt.close(figure)

    def _save_uncertainty_distribution(
        self,
        uncertainty_level: np.ndarray,
        output_path: Path,
    ) -> None:
        labels = ["Low", "Medium", "High"]
        counts = [
            int(np.sum(uncertainty_level == label))
            for label in labels
        ]

        figure, axis = plt.subplots(figsize=(8, 6))
        axis.bar(labels, counts)
        axis.set_xlabel("Uncertainty level")
        axis.set_ylabel("Number of samples")
        axis.set_title("Uncertainty Level Distribution")

        for index, count in enumerate(counts):
            axis.text(
                index,
                count,
                str(count),
                ha="center",
                va="bottom",
            )

        figure.tight_layout()
        figure.savefig(output_path, dpi=250, bbox_inches="tight")
        plt.close(figure)

    def _save_confidence_vs_uncertainty(
        self,
        confidence: np.ndarray,
        normalized_entropy: np.ndarray,
        correctness: np.ndarray,
        output_path: Path,
    ) -> None:
        figure, axis = plt.subplots(figsize=(9, 7))

        correct_mask = correctness == 1
        incorrect_mask = correctness == 0

        axis.scatter(
            confidence[correct_mask],
            normalized_entropy[correct_mask],
            alpha=0.25,
            s=12,
            label="Correct",
        )
        axis.scatter(
            confidence[incorrect_mask],
            normalized_entropy[incorrect_mask],
            alpha=0.35,
            s=16,
            label="Incorrect",
        )

        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("Mean prediction confidence")
        axis.set_ylabel("Normalized predictive entropy")
        axis.set_title("Confidence vs Uncertainty")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path, dpi=250, bbox_inches="tight")
        plt.close(figure)

    def _save_json(
        self,
        payload: dict[str, Any],
        output_path: Path,
    ) -> None:
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    def _save_summary(
        self,
        metrics: dict[str, Any],
        output_path: Path,
    ) -> None:
        summary = "\n".join(
            [
                "Monte Carlo Dropout Uncertainty Evaluation",
                "=" * 43,
                "",
                f"Checkpoint: {metrics['checkpoint']}",
                f"Round: {metrics['round']}",
                f"Samples: {metrics['samples']}",
                f"MC samples: {metrics['mc_samples']}",
                f"Dropout layers: {metrics['dropout_layers']}",
                "",
                f"Accuracy: {metrics['accuracy']:.6f}",
                f"Average confidence: {metrics['average_confidence']:.6f}",
                (
                    "Mean predictive entropy: "
                    f"{metrics['mean_predictive_entropy']:.6f}"
                ),
                (
                    "Mean normalized entropy: "
                    f"{metrics['mean_normalized_entropy']:.6f}"
                ),
                (
                    "Mean mutual information: "
                    f"{metrics['mean_mutual_information']:.6f}"
                ),
                (
                    "Mean predictive variance: "
                    f"{metrics['mean_predictive_variance']:.6f}"
                ),
                (
                    "Confidence-uncertainty correlation: "
                    f"{metrics['confidence_uncertainty_correlation']:.6f}"
                ),
                "",
                "Uncertainty counts",
                f"Low: {metrics['uncertainty_counts']['Low']}",
                f"Medium: {metrics['uncertainty_counts']['Medium']}",
                f"High: {metrics['uncertainty_counts']['High']}",
                "",
                "Accuracy by uncertainty level",
                (
                    "Low: "
                    f"{metrics['accuracy_by_uncertainty_level']['Low']}"
                ),
                (
                    "Medium: "
                    f"{metrics['accuracy_by_uncertainty_level']['Medium']}"
                ),
                (
                    "High: "
                    f"{metrics['accuracy_by_uncertainty_level']['High']}"
                ),
            ]
        )

        output_path.write_text(summary, encoding="utf-8")
