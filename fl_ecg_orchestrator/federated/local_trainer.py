from __future__ import annotations

from typing import Any

import torch

from fl_ecg_orchestrator.config.project_config import (
    ProjectConfig,
)
from fl_ecg_orchestrator.data.loader import (
    load_client_data,
)
from fl_ecg_orchestrator.federated.training import (
    evaluate_model,
    train_local_model,
)
from fl_ecg_orchestrator.model.cnn1d import (
    create_model,
)


class LocalTrainer:
    def __init__(
        self,
        client_id: str,
        config: ProjectConfig,
        use_smotetomek: bool,
        proximal_mu: float,
        seed: int,
    ) -> None:
        self.client_id = client_id
        self.config = config
        self.use_smotetomek = bool(
            use_smotetomek
        )
        self.proximal_mu = float(
            proximal_mu
        )
        self.seed = int(seed)

        self.model = create_model(
            input_length=(
                config.data.input_length
            ),
            num_classes=(
                config.data.num_classes
            ),
        )

        self.client_data = load_client_data(
            client_id=client_id,
            config_path=config.source_path,
            use_smotetomek=(
                self.use_smotetomek
            ),
        )

    def train(
        self,
        local_epochs: int,
        learning_rate: float,
        proximal_mu: float | None = None,
    ) -> dict[str, Any]:
        effective_mu = (
            self.proximal_mu
            if proximal_mu is None
            else float(proximal_mu)
        )

        result = train_local_model(
            model=self.model,
            train_loader=self.client_data[
                "train_loader"
            ],
            validation_loader=self.client_data[
                "validation_loader"
            ],
            local_epochs=int(
                local_epochs
            ),
            learning_rate=float(
                learning_rate
            ),
            proximal_mu=effective_mu,
            seed=self.seed,
        )

        self.model = result["model"]

        return result

    def evaluate(
        self,
    ) -> dict[str, Any]:
        return evaluate_model(
            model=self.model,
            loader=self.client_data[
                "validation_loader"
            ],
        )

    def get_training_sample_count(
        self,
    ) -> int:
        return int(
            len(
                self.client_data[
                    "train_loader"
                ].dataset
            )
        )

    def get_validation_sample_count(
        self,
    ) -> int:
        return int(
            len(
                self.client_data[
                    "validation_loader"
                ].dataset
            )
        )

    def get_metadata(
        self,
    ) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "training_records": list(
                self.client_data[
                    "training_records"
                ]
            ),
            "validation_records": list(
                self.client_data[
                    "validation_records"
                ]
            ),
            "training_samples": (
                self.get_training_sample_count()
            ),
            "validation_samples": (
                self.get_validation_sample_count()
            ),
            "training_distribution": (
                self.client_data[
                    "training_distribution"
                ]
            ),
            "validation_distribution": (
                self.client_data[
                    "validation_distribution"
                ]
            ),
            "smotetomek_enabled": (
                self.use_smotetomek
            ),
            "balancing_metadata": (
                self.client_data[
                    "balancing_metadata"
                ]
            ),
            "proximal_mu": (
                self.proximal_mu
            ),
            "seed": self.seed,
            "device": str(
                next(
                    self.model.parameters()
                ).device
            ),
            "torch_version": (
                torch.__version__
            ),
        }