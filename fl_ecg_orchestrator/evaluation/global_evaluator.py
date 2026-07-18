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
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from fl_ecg_orchestrator.data.loader import (
    load_config,
    load_global_test_data,
)
from fl_ecg_orchestrator.federated.training import evaluate_model
from fl_ecg_orchestrator.model.cnn1d import create_model


CLASS_NAMES = {
    0: "N",
    1: "S",
    2: "V",
    3: "F",
    4: "Q",
}


class GlobalEvaluator:

    def __init__(
        self,
        config_path: str = (
            "fl_ecg_orchestrator/config/config.yaml"
        ),
        device: str | None = None,
    ):
        self.config_path = config_path
        self.config = load_config(config_path)

        self.device = torch.device(
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.test_data = load_global_test_data(
            config_path
        )

        self.model = create_model(
            input_length=int(
                self.config["data"]["input_length"]
            ),
            num_classes=int(
                self.config["data"]["num_classes"]
            ),
        ).to(self.device)

    def evaluate_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, Any]:

        self.model.load_state_dict(
            state_dict,
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

        return evaluate_model(
            model=self.model,
            loader=self.test_data["loader"],
        )

    @torch.no_grad()
    def collect_predictions(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:

        self.model.eval()

        all_labels: list[np.ndarray] = []
        all_predictions: list[np.ndarray] = []
        all_probabilities: list[np.ndarray] = []

        for batch in self.test_data["loader"]:

            if len(batch) < 2:
                raise ValueError(
                    "Global test loader must return "
                    "features and labels."
                )

            features = batch[0].to(
                self.device,
                dtype=torch.float32,
            )

            labels = batch[1].to(
                self.device,
                dtype=torch.long,
            )

            logits = self.model(features)

            probabilities = torch.softmax(
                logits,
                dim=1,
            ).to(torch.float64)

            probabilities = probabilities / probabilities.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1e-12)

            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

            all_labels.append(
                labels.detach().cpu().numpy()
            )

            all_predictions.append(
                predictions.detach().cpu().numpy()
            )

            all_probabilities.append(
                probabilities.detach().cpu().numpy()
            )

        if not all_labels:
            raise RuntimeError(
                "No samples were returned by the "
                "global test loader."
            )

        true_labels = np.concatenate(
            all_labels
        ).astype(np.int64)

        predictions = np.concatenate(
            all_predictions
        ).astype(np.int64)

        probabilities = np.concatenate(
            all_probabilities
        ).astype(np.float64)

        probability_sums = probabilities.sum(
            axis=1,
            keepdims=True,
        )

        probabilities = np.divide(
            probabilities,
            probability_sums,
            out=np.zeros_like(probabilities),
            where=probability_sums > 0,
        )

        return (
            true_labels,
            predictions,
            probabilities,
        )

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
    ) -> dict[str, Any]:

        resolved_path = Path(
            checkpoint_path
        ).resolve()

        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: "
                f"{resolved_path}"
            )

        checkpoint = torch.load(
            resolved_path,
            map_location=self.device,
            weights_only=False,
        )

        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint,
        )

        self.model.load_state_dict(
            state_dict,
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

        return checkpoint

    def advanced_evaluation(
        self,
        checkpoint_path: str | Path,
        output_dir: str | Path = (
            "outputs/evaluation"
        ),
    ) -> dict[str, Any]:

        checkpoint = self.load_checkpoint(
            checkpoint_path
        )

        output_path = Path(
            output_dir
        ).resolve()

        output_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            true_labels,
            predictions,
            probabilities,
        ) = self.collect_predictions()

        class_ids = list(
            range(
                int(
                    self.config[
                        "data"
                    ]["num_classes"]
                )
            )
        )

        class_names = [
            CLASS_NAMES.get(
                class_id,
                str(class_id),
            )
            for class_id in class_ids
        ]

        metrics = self._calculate_metrics(
            true_labels=true_labels,
            predictions=predictions,
            probabilities=probabilities,
            class_ids=class_ids,
            class_names=class_names,
        )

        matrix = confusion_matrix(
            true_labels,
            predictions,
            labels=class_ids,
        )

        self._save_confusion_matrix(
            matrix=matrix,
            class_names=class_names,
            output_path=(
                output_path
                / "confusion_matrix.png"
            ),
            normalized=False,
        )

        self._save_confusion_matrix(
            matrix=matrix,
            class_names=class_names,
            output_path=(
                output_path
                / "confusion_matrix_normalized.png"
            ),
            normalized=True,
        )

        auc_scores = self._save_roc_curves(
            true_labels=true_labels,
            probabilities=probabilities,
            class_ids=class_ids,
            class_names=class_names,
            output_path=(
                output_path
                / "roc_curve_multiclass.png"
            ),
        )

        report_dictionary = (
            classification_report(
                true_labels,
                predictions,
                labels=class_ids,
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )
        )

        report_text = classification_report(
            true_labels,
            predictions,
            labels=class_ids,
            target_names=class_names,
            zero_division=0,
        )

        self._save_json(
            output_path / "metrics.json",
            metrics,
        )

        self._save_json(
            output_path
            / "classification_report.json",
            report_dictionary,
        )

        self._save_classification_report_csv(
            report_dictionary,
            output_path
            / "classification_report.csv",
        )

        self._save_per_class_metrics_csv(
            metrics["per_class"],
            output_path
            / "per_class_metrics.csv",
        )

        self._save_auc_csv(
            auc_scores,
            output_path / "auc_scores.csv",
        )

        np.savez_compressed(
            output_path
            / "evaluation_predictions.npz",
            true_labels=true_labels,
            predictions=predictions,
            probabilities=probabilities,
            confusion_matrix=matrix,
        )

        summary = self._build_summary(
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            metrics=metrics,
            report_text=report_text,
        )

        (
            output_path
            / "evaluation_summary.txt"
        ).write_text(
            summary,
            encoding="utf-8",
        )

        return {
            "checkpoint": str(
                Path(
                    checkpoint_path
                ).resolve()
            ),
            "output_dir": str(
                output_path
            ),
            "metrics": metrics,
            "auc_scores": auc_scores,
            "samples": int(
                len(true_labels)
            ),
        }

    def _calculate_metrics(
        self,
        true_labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        class_ids: list[int],
        class_names: list[str],
    ) -> dict[str, Any]:

        per_class_report = (
            classification_report(
                true_labels,
                predictions,
                labels=class_ids,
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )
        )

        per_class_auc = (
            self._calculate_per_class_auc(
                true_labels=true_labels,
                probabilities=probabilities,
                class_ids=class_ids,
                class_names=class_names,
            )
        )

        per_class_metrics: dict[
            str,
            dict[str, Any],
        ] = {}

        for class_id, class_name in zip(
            class_ids,
            class_names,
        ):

            values = per_class_report[
                class_name
            ]

            class_mask = (
                true_labels == class_id
            )

            class_accuracy = (
                float(
                    np.mean(
                        predictions[
                            class_mask
                        ]
                        == class_id
                    )
                )
                if np.any(class_mask)
                else None
            )

            per_class_metrics[
                class_name
            ] = {
                "class_id": int(
                    class_id
                ),
                "precision": float(
                    values["precision"]
                ),
                "recall": float(
                    values["recall"]
                ),
                "f1_score": float(
                    values["f1-score"]
                ),
                "support": int(
                    values["support"]
                ),
                "class_accuracy": (
                    class_accuracy
                ),
                "roc_auc": (
                    per_class_auc.get(
                        class_name
                    )
                ),
            }

        metrics: dict[str, Any] = {
            "samples": int(
                len(true_labels)
            ),
            "accuracy": float(
                accuracy_score(
                    true_labels,
                    predictions,
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    true_labels,
                    predictions,
                )
            ),
            "macro_precision": float(
                precision_score(
                    true_labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_recall": float(
                recall_score(
                    true_labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_f1": float(
                f1_score(
                    true_labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
            "weighted_precision": float(
                precision_score(
                    true_labels,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "weighted_recall": float(
                recall_score(
                    true_labels,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "weighted_f1": float(
                f1_score(
                    true_labels,
                    predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "matthews_correlation_coefficient": (
                float(
                    matthews_corrcoef(
                        true_labels,
                        predictions,
                    )
                )
            ),
            "log_loss": float(
                log_loss(
                    true_labels,
                    probabilities,
                    labels=class_ids,
                )
            ),
            "per_class": per_class_metrics,
        }

        try:
            metrics["macro_roc_auc"] = float(
                roc_auc_score(
                    true_labels,
                    probabilities,
                    labels=class_ids,
                    multi_class="ovr",
                    average="macro",
                )
            )
        except ValueError:
            metrics[
                "macro_roc_auc"
            ] = None

        try:
            metrics[
                "weighted_roc_auc"
            ] = float(
                roc_auc_score(
                    true_labels,
                    probabilities,
                    labels=class_ids,
                    multi_class="ovr",
                    average="weighted",
                )
            )
        except ValueError:
            metrics[
                "weighted_roc_auc"
            ] = None

        return metrics

    def _calculate_per_class_auc(
        self,
        true_labels: np.ndarray,
        probabilities: np.ndarray,
        class_ids: list[int],
        class_names: list[str],
    ) -> dict[str, float | None]:

        binary_labels = label_binarize(
            true_labels,
            classes=class_ids,
        )

        results: dict[
            str,
            float | None,
        ] = {}

        for position, class_name in enumerate(
            class_names
        ):

            targets = binary_labels[
                :,
                position,
            ]

            if (
                np.unique(
                    targets
                ).size
                < 2
            ):
                results[
                    class_name
                ] = None
                continue

            results[
                class_name
            ] = float(
                roc_auc_score(
                    targets,
                    probabilities[
                        :,
                        position,
                    ],
                )
            )

        return results

    def _save_confusion_matrix(
        self,
        matrix: np.ndarray,
        class_names: list[str],
        output_path: Path,
        normalized: bool,
    ) -> None:

        display_matrix = (
            matrix.astype(
                np.float64
            )
        )

        if normalized:
            row_totals = (
                display_matrix.sum(
                    axis=1,
                    keepdims=True,
                )
            )

            display_matrix = np.divide(
                display_matrix,
                row_totals,
                out=np.zeros_like(
                    display_matrix
                ),
                where=row_totals != 0,
            )

        figure, axis = plt.subplots(
            figsize=(8, 7)
        )

        image = axis.imshow(
            display_matrix,
            interpolation="nearest",
        )

        figure.colorbar(
            image,
            ax=axis,
        )

        axis.set(
            xticks=np.arange(
                len(class_names)
            ),
            yticks=np.arange(
                len(class_names)
            ),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel="Predicted class",
            ylabel="True class",
            title=(
                "Normalized Confusion Matrix"
                if normalized
                else "Confusion Matrix"
            ),
        )

        threshold = (
            float(
                display_matrix.max()
            )
            / 2
            if display_matrix.size
            else 0
        )

        for row in range(
            display_matrix.shape[0]
        ):
            for column in range(
                display_matrix.shape[1]
            ):

                value = display_matrix[
                    row,
                    column,
                ]

                text = (
                    f"{value:.2f}"
                    if normalized
                    else str(
                        int(value)
                    )
                )

                axis.text(
                    column,
                    row,
                    text,
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if value > threshold
                        else "black"
                    ),
                )

        figure.tight_layout()

        figure.savefig(
            output_path,
            dpi=250,
            bbox_inches="tight",
        )

        plt.close(
            figure
        )

    def _save_roc_curves(
        self,
        true_labels: np.ndarray,
        probabilities: np.ndarray,
        class_ids: list[int],
        class_names: list[str],
        output_path: Path,
    ) -> dict[str, float | None]:

        binary_labels = label_binarize(
            true_labels,
            classes=class_ids,
        )

        auc_scores: dict[
            str,
            float | None,
        ] = {}

        figure, axis = plt.subplots(
            figsize=(9, 8)
        )

        valid_fpr: list[np.ndarray] = []
        valid_tpr: list[np.ndarray] = []

        for position, class_name in enumerate(
            class_names
        ):

            targets = binary_labels[
                :,
                position,
            ]

            if (
                np.unique(
                    targets
                ).size
                < 2
            ):
                auc_scores[
                    class_name
                ] = None
                continue

            false_positive_rate, (
                true_positive_rate
            ), _ = roc_curve(
                targets,
                probabilities[
                    :,
                    position,
                ],
            )

            score = float(
                roc_auc_score(
                    targets,
                    probabilities[
                        :,
                        position,
                    ],
                )
            )

            auc_scores[
                class_name
            ] = score

            valid_fpr.append(
                false_positive_rate
            )

            valid_tpr.append(
                true_positive_rate
            )

            axis.plot(
                false_positive_rate,
                true_positive_rate,
                linewidth=2,
                label=(
                    f"{class_name} "
                    f"(AUC={score:.3f})"
                ),
            )

        flattened_targets = (
            binary_labels.ravel()
        )

        flattened_probabilities = (
            probabilities.ravel()
        )

        if (
            np.unique(
                flattened_targets
            ).size
            == 2
        ):
            micro_fpr, (
                micro_tpr
            ), _ = roc_curve(
                flattened_targets,
                flattened_probabilities,
            )

            micro_auc = float(
                roc_auc_score(
                    flattened_targets,
                    flattened_probabilities,
                )
            )

            auc_scores[
                "micro_average"
            ] = micro_auc

            axis.plot(
                micro_fpr,
                micro_tpr,
                linestyle="--",
                linewidth=2.5,
                label=(
                    "Micro-average "
                    f"(AUC={micro_auc:.3f})"
                ),
            )
        else:
            auc_scores[
                "micro_average"
            ] = None

        if valid_fpr:
            combined_fpr = np.unique(
                np.concatenate(
                    valid_fpr
                )
            )

            mean_tpr = np.zeros_like(
                combined_fpr
            )

            for fpr_values, tpr_values in zip(
                valid_fpr,
                valid_tpr,
            ):
                mean_tpr += np.interp(
                    combined_fpr,
                    fpr_values,
                    tpr_values,
                )

            mean_tpr /= len(
                valid_fpr
            )

            macro_auc = float(
                np.trapezoid(
                    mean_tpr,
                    combined_fpr,
                )
            )

            auc_scores[
                "macro_average"
            ] = macro_auc

            axis.plot(
                combined_fpr,
                mean_tpr,
                linestyle=":",
                linewidth=3,
                label=(
                    "Macro-average "
                    f"(AUC={macro_auc:.3f})"
                ),
            )
        else:
            auc_scores[
                "macro_average"
            ] = None

        axis.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=1,
        )

        axis.set_xlim(
            0,
            1,
        )

        axis.set_ylim(
            0,
            1.02,
        )

        axis.set_xlabel(
            "False Positive Rate"
        )

        axis.set_ylabel(
            "True Positive Rate"
        )

        axis.set_title(
            "Multiclass ROC Curves"
        )

        axis.legend(
            loc="lower right"
        )

        figure.tight_layout()

        figure.savefig(
            output_path,
            dpi=250,
            bbox_inches="tight",
        )

        plt.close(
            figure
        )

        return auc_scores

    def _save_classification_report_csv(
        self,
        report: dict[str, Any],
        output_path: Path,
    ) -> None:

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow(
                [
                    "class",
                    "precision",
                    "recall",
                    "f1_score",
                    "support",
                ]
            )

            for label, values in (
                report.items()
            ):

                if not isinstance(
                    values,
                    dict,
                ):
                    continue

                writer.writerow(
                    [
                        label,
                        values.get(
                            "precision"
                        ),
                        values.get(
                            "recall"
                        ),
                        values.get(
                            "f1-score"
                        ),
                        values.get(
                            "support"
                        ),
                    ]
                )

    def _save_per_class_metrics_csv(
        self,
        metrics: dict[str, Any],
        output_path: Path,
    ) -> None:

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(
                file
            )

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

            for class_name, values in (
                metrics.items()
            ):
                writer.writerow(
                    [
                        class_name,
                        values[
                            "class_id"
                        ],
                        values[
                            "precision"
                        ],
                        values[
                            "recall"
                        ],
                        values[
                            "f1_score"
                        ],
                        values[
                            "support"
                        ],
                        values[
                            "class_accuracy"
                        ],
                        values[
                            "roc_auc"
                        ],
                    ]
                )

    def _save_auc_csv(
        self,
        auc_scores: dict[
            str,
            float | None,
        ],
        output_path: Path,
    ) -> None:

        with output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow(
                [
                    "curve",
                    "auc",
                ]
            )

            for label, score in (
                auc_scores.items()
            ):
                writer.writerow(
                    [
                        label,
                        score,
                    ]
                )

    def _save_json(
        self,
        output_path: Path,
        payload: dict[str, Any],
    ) -> None:

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
            )

    def _build_summary(
        self,
        checkpoint_path: str | Path,
        checkpoint: dict[str, Any],
        metrics: dict[str, Any],
        report_text: str,
    ) -> str:

        return "\n".join(
            [
                "Advanced ECG Model Evaluation",
                "=" * 29,
                "",
                (
                    "Checkpoint: "
                    f"{Path(checkpoint_path).resolve()}"
                ),
                f"Device: {self.device}",
                (
                    "Round: "
                    f"{checkpoint.get('round')}"
                ),
                (
                    "Samples: "
                    f"{metrics['samples']}"
                ),
                "",
                "Overall metrics",
                "-" * 15,
                (
                    "Accuracy: "
                    f"{metrics['accuracy']:.6f}"
                ),
                (
                    "Balanced accuracy: "
                    f"{metrics['balanced_accuracy']:.6f}"
                ),
                (
                    "Macro precision: "
                    f"{metrics['macro_precision']:.6f}"
                ),
                (
                    "Macro recall: "
                    f"{metrics['macro_recall']:.6f}"
                ),
                (
                    "Macro F1: "
                    f"{metrics['macro_f1']:.6f}"
                ),
                (
                    "Weighted F1: "
                    f"{metrics['weighted_f1']:.6f}"
                ),
                (
                    "MCC: "
                    f"{metrics['matthews_correlation_coefficient']:.6f}"
                ),
                (
                    "Log loss: "
                    f"{metrics['log_loss']:.6f}"
                ),
                (
                    "Macro ROC AUC: "
                    f"{metrics['macro_roc_auc']}"
                ),
                (
                    "Weighted ROC AUC: "
                    f"{metrics['weighted_roc_auc']}"
                ),
                "",
                "Classification report",
                "-" * 21,
                report_text,
            ]
        )

    def save_checkpoint(
        self,
        state_dict: dict[str, torch.Tensor],
        experiment_name: str,
        round_number: int,
        metrics: dict[str, Any],
    ) -> Path:

        output_dir = Path(
            self.config["project"]["output_dir"]
        ) / "checkpoints"

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint_path = output_dir / (
            f"{experiment_name}_"
            f"round_{round_number}.pth"
        )

        torch.save(
            {
                "model_state_dict": state_dict,
                "round": int(
                    round_number
                ),
                "metrics": metrics,
                "input_length": int(
                    self.config[
                        "data"
                    ]["input_length"]
                ),
                "num_classes": int(
                    self.config[
                        "data"
                    ]["num_classes"]
                ),
                "global_test_records": (
                    self.test_data[
                        "record_ids"
                    ]
                ),
            },
            checkpoint_path,
        )

        return checkpoint_path