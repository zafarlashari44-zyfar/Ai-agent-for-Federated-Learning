from pathlib import Path
from typing import Any

from flwr.app import ArrayRecord, MetricRecord

from fl_ecg_orchestrator.evaluation.experiment_logger import (
    ExperimentLogger,
)
from fl_ecg_orchestrator.evaluation.global_evaluator import (
    GlobalEvaluator,
)


class CentralEvaluator:

    def __init__(
        self,
        evaluator: GlobalEvaluator,
        logger: ExperimentLogger,
        experiment_name: str,
    ) -> None:
        self.evaluator = evaluator
        self.logger = logger
        self.experiment_name = experiment_name
        self.best_macro_f1 = float("-inf")
        self.best_round = 0
        self.best_checkpoint_path: Path | None = None

    def __call__(
        self,
        server_round: int,
        arrays: ArrayRecord,
    ) -> MetricRecord:
        state_dict = arrays.to_torch_state_dict()

        metrics = self.evaluator.evaluate_state_dict(
            state_dict
        )

        macro_f1 = float(metrics["macro_f1"])

        checkpoint_path = None

        if macro_f1 > self.best_macro_f1:
            self.best_macro_f1 = macro_f1
            self.best_round = int(server_round)

            checkpoint_path = (
                self.evaluator.save_checkpoint(
                    state_dict=state_dict,
                    experiment_name=(
                        f"{self.experiment_name}_best"
                    ),
                    round_number=server_round,
                    metrics=metrics,
                )
            )

            self.best_checkpoint_path = (
                checkpoint_path
            )

        round_payload = {
            "round": int(server_round),
            "global_test_metrics": (
                self._make_json_safe(metrics)
            ),
            "is_best_round": (
                int(server_round)
                == self.best_round
            ),
            "best_macro_f1": float(
                self.best_macro_f1
            ),
            "checkpoint_path": (
                str(checkpoint_path)
                if checkpoint_path
                else None
            ),
        }

        self.logger.write_json(
            filename=(
                f"global_round_{server_round}.json"
            ),
            payload=round_payload,
        )

        self._print_round_summary(
            server_round,
            metrics,
        )

        return MetricRecord(
            {
                "num-examples": int(
                    metrics["num_samples"]
                ),
                "loss": float(
                    metrics["loss"]
                ),
                "accuracy": float(
                    metrics["accuracy"]
                ),
                "macro-precision": float(
                    metrics["macro_precision"]
                ),
                "macro-recall": float(
                    metrics["macro_recall"]
                ),
                "macro-f1": float(
                    metrics["macro_f1"]
                ),
                "recall-N": float(
                    metrics[
                        "per_class_recall"
                    ]["N"]
                ),
                "recall-S": float(
                    metrics[
                        "per_class_recall"
                    ]["S"]
                ),
                "recall-V": float(
                    metrics[
                        "per_class_recall"
                    ]["V"]
                ),
                "recall-F": float(
                    metrics[
                        "per_class_recall"
                    ]["F"]
                ),
                "recall-Q": float(
                    metrics[
                        "per_class_recall"
                    ]["Q"]
                ),
                "f1-N": float(
                    metrics[
                        "per_class_f1"
                    ]["N"]
                ),
                "f1-S": float(
                    metrics[
                        "per_class_f1"
                    ]["S"]
                ),
                "f1-V": float(
                    metrics[
                        "per_class_f1"
                    ]["V"]
                ),
                "f1-F": float(
                    metrics[
                        "per_class_f1"
                    ]["F"]
                ),
                "f1-Q": float(
                    metrics[
                        "per_class_f1"
                    ]["Q"]
                ),
            }
        )

    def get_summary(self) -> dict[str, Any]:
        return {
            "best_round": int(
                self.best_round
            ),
            "best_macro_f1": float(
                self.best_macro_f1
            ),
            "best_checkpoint_path": (
                str(self.best_checkpoint_path)
                if self.best_checkpoint_path
                else None
            ),
        }

    @staticmethod
    def _make_json_safe(
        value: Any,
    ) -> Any:
        if isinstance(value, dict):
            return {
                key: CentralEvaluator._make_json_safe(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [
                CentralEvaluator._make_json_safe(
                    item
                )
                for item in value
            ]

        if hasattr(value, "item"):
            return value.item()

        return value

    @staticmethod
    def _print_round_summary(
        server_round: int,
        metrics: dict[str, Any],
    ) -> None:
        recall = metrics[
            "per_class_recall"
        ]

        print()
        print(
            f"Global round {server_round}"
        )
        print(
            f"Accuracy {metrics['accuracy']:.4f}"
        )
        print(
            f"Macro F1 {metrics['macro_f1']:.4f}"
        )
        print(
            f"Recall N {recall['N']:.4f}"
        )
        print(
            f"Recall S {recall['S']:.4f}"
        )
        print(
            f"Recall V {recall['V']:.4f}"
        )
        print(
            f"Recall F {recall['F']:.4f}"
        )
        print(
            f"Recall Q {recall['Q']:.4f}"
        )