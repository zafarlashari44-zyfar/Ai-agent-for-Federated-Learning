from pathlib import Path
from typing import Any

import torch

from fl_ecg_orchestrator.data.loader import (
    load_config,
    load_global_test_data,
)
from fl_ecg_orchestrator.federated.training import evaluate_model
from fl_ecg_orchestrator.model.cnn1d import create_model


class GlobalEvaluator:

    def __init__(
        self,
        config_path: str = "fl_ecg_orchestrator/config/config.yaml",
    ):
        self.config_path = config_path
        self.config = load_config(config_path)

        self.test_data = load_global_test_data(config_path)

        self.model = create_model(
            input_length=int(self.config["data"]["input_length"]),
            num_classes=int(self.config["data"]["num_classes"]),
        )

    def evaluate_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, Any]:

        self.model.load_state_dict(
            state_dict,
            strict=True,
        )

        return evaluate_model(
            model=self.model,
            loader=self.test_data["loader"],
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
            f"{experiment_name}_round_{round_number}.pth"
        )

        torch.save(
            {
                "model_state_dict": state_dict,
                "round": int(round_number),
                "metrics": metrics,
                "input_length": int(
                    self.config["data"]["input_length"]
                ),
                "num_classes": int(
                    self.config["data"]["num_classes"]
                ),
                "global_test_records": self.test_data[
                    "record_ids"
                ],
            },
            checkpoint_path,
        )

        return checkpoint_path