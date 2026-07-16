from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "fl_ecg_orchestrator"
    / "config"
    / "config.yaml"
)


def resolve_project_path(
    path_value: str | Path,
) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class ProjectSettings:
    name: str
    seed: int
    records_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class DataSettings:
    input_length: int
    num_classes: int
    label_map: dict[str, int]
    global_test_records: tuple[str, ...]


@dataclass(frozen=True)
class TrainingSettings:
    federated_rounds: int
    local_epochs: int
    batch_size: int
    learning_rate: float
    validation_ratio: float


@dataclass(frozen=True)
class StrategySettings:
    name: str
    proximal_mu: float
    smotetomek: bool


@dataclass(frozen=True)
class ExperimentDefinition:
    name: str
    strategy: str
    smotetomek: bool
    proximal_mu: float | None
    proximal_mu_values: tuple[float, ...]


@dataclass(frozen=True)
class AblationSettings:
    seeds: tuple[int, ...]
    experiments: tuple[ExperimentDefinition, ...]


@dataclass(frozen=True)
class ProjectConfig:
    project: ProjectSettings
    data: DataSettings
    training: TrainingSettings
    strategy: StrategySettings
    clients: dict[str, tuple[str, ...]]
    ablation: AblationSettings
    source_path: Path

    @classmethod
    def load(
        cls,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> "ProjectConfig":
        resolved_path = resolve_project_path(
            config_path
        )

        if not resolved_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at "
                f"{resolved_path}"
            )

        with resolved_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw_config = yaml.safe_load(file)

        if not isinstance(raw_config, dict):
            raise ValueError(
                "Configuration must contain a YAML mapping"
            )

        cls._validate_required_sections(
            raw_config
        )

        project_raw = raw_config["project"]
        data_raw = raw_config["data"]
        training_raw = raw_config["training"]
        strategy_raw = raw_config["strategy"]
        clients_raw = raw_config["clients"]
        ablation_raw = raw_config.get(
            "ablation",
            {
                "seeds": [
                    project_raw["seed"]
                ],
                "experiments": [],
            },
        )

        config = cls(
            project=ProjectSettings(
                name=str(
                    project_raw["name"]
                ),
                seed=int(
                    project_raw["seed"]
                ),
                records_dir=resolve_project_path(
                    project_raw["records_dir"]
                ),
                output_dir=resolve_project_path(
                    project_raw["output_dir"]
                ),
            ),
            data=DataSettings(
                input_length=int(
                    data_raw["input_length"]
                ),
                num_classes=int(
                    data_raw["num_classes"]
                ),
                label_map={
                    str(label): int(index)
                    for label, index
                    in data_raw["label_map"].items()
                },
                global_test_records=tuple(
                    str(record_id)
                    for record_id
                    in data_raw[
                        "global_test_records"
                    ]
                ),
            ),
            training=TrainingSettings(
                federated_rounds=int(
                    training_raw[
                        "federated_rounds"
                    ]
                ),
                local_epochs=int(
                    training_raw[
                        "local_epochs"
                    ]
                ),
                batch_size=int(
                    training_raw[
                        "batch_size"
                    ]
                ),
                learning_rate=float(
                    training_raw[
                        "learning_rate"
                    ]
                ),
                validation_ratio=float(
                    training_raw[
                        "validation_ratio"
                    ]
                ),
            ),
            strategy=StrategySettings(
                name=str(
                    strategy_raw["name"]
                ).lower(),
                proximal_mu=float(
                    strategy_raw[
                        "proximal_mu"
                    ]
                ),
                smotetomek=bool(
                    strategy_raw[
                        "smotetomek"
                    ]
                ),
            ),
            clients={
                str(client_id): tuple(
                    str(record_id)
                    for record_id
                    in record_ids
                )
                for client_id, record_ids
                in clients_raw.items()
            },
            ablation=AblationSettings(
                seeds=tuple(
                    int(seed)
                    for seed
                    in ablation_raw.get(
                        "seeds",
                        [],
                    )
                ),
                experiments=tuple(
                    cls._parse_experiment(
                        experiment
                    )
                    for experiment
                    in ablation_raw.get(
                        "experiments",
                        [],
                    )
                ),
            ),
            source_path=resolved_path,
        )

        config.validate()

        return config

    @staticmethod
    def _validate_required_sections(
        raw_config: dict[str, Any],
    ) -> None:
        required_sections = {
            "project",
            "data",
            "clients",
            "training",
            "strategy",
        }

        missing_sections = (
            required_sections
            - set(raw_config)
        )

        if missing_sections:
            raise KeyError(
                f"Missing configuration sections "
                f"{sorted(missing_sections)}"
            )

    @staticmethod
    def _parse_experiment(
        experiment: dict[str, Any],
    ) -> ExperimentDefinition:
        proximal_mu = experiment.get(
            "proximal_mu"
        )

        proximal_mu_values = experiment.get(
            "proximal_mu_values",
            [],
        )

        return ExperimentDefinition(
            name=str(
                experiment["name"]
            ),
            strategy=str(
                experiment["strategy"]
            ).lower(),
            smotetomek=bool(
                experiment.get(
                    "smotetomek",
                    False,
                )
            ),
            proximal_mu=(
                float(proximal_mu)
                if proximal_mu is not None
                else None
            ),
            proximal_mu_values=tuple(
                float(value)
                for value
                in proximal_mu_values
            ),
        )

    def validate(self) -> None:
        if not self.project.records_dir.exists():
            raise FileNotFoundError(
                f"Records directory not found at "
                f"{self.project.records_dir}"
            )

        if self.data.input_length <= 0:
            raise ValueError(
                "input_length must be greater than zero"
            )

        if self.data.num_classes <= 1:
            raise ValueError(
                "num_classes must be greater than one"
            )

        if len(self.data.label_map) != (
            self.data.num_classes
        ):
            raise ValueError(
                "label_map size does not match "
                "num_classes"
            )

        if self.training.federated_rounds <= 0:
            raise ValueError(
                "federated_rounds must be greater "
                "than zero"
            )

        if self.training.local_epochs <= 0:
            raise ValueError(
                "local_epochs must be greater than zero"
            )

        if self.training.batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero"
            )

        if self.training.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than zero"
            )

        if not (
            0
            < self.training.validation_ratio
            < 1
        ):
            raise ValueError(
                "validation_ratio must be between "
                "zero and one"
            )

        if self.strategy.name not in {
            "fedavg",
            "fedprox",
        }:
            raise ValueError(
                f"Unsupported strategy "
                f"{self.strategy.name}"
            )

        if (
            self.strategy.name == "fedprox"
            and self.strategy.proximal_mu <= 0
        ):
            raise ValueError(
                "FedProx requires proximal_mu "
                "greater than zero"
            )

        if not self.clients:
            raise ValueError(
                "At least one federated client "
                "is required"
            )

        all_client_records = [
            record_id
            for records in self.clients.values()
            for record_id in records
        ]

        if len(all_client_records) != len(
            set(all_client_records)
        ):
            raise ValueError(
                "A patient record appears in "
                "multiple clients"
            )

        leaked_records = (
            set(all_client_records)
            & set(
                self.data.global_test_records
            )
        )

        if leaked_records:
            raise ValueError(
                f"Global test records leaked into "
                f"clients {sorted(leaked_records)}"
            )

    def client_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                self.clients,
                key=lambda value: int(
                    value.split("_")[-1]
                ),
            )
        )

    def client_records(
        self,
        client_id: str,
    ) -> tuple[str, ...]:
        if client_id not in self.clients:
            raise KeyError(
                f"Unknown client {client_id}"
            )

        return self.clients[client_id]

    def summary(self) -> dict[str, Any]:
        return {
            "project_name": self.project.name,
            "config_path": str(
                self.source_path
            ),
            "records_dir": str(
                self.project.records_dir
            ),
            "output_dir": str(
                self.project.output_dir
            ),
            "seed": self.project.seed,
            "num_clients": len(
                self.clients
            ),
            "num_global_test_records": len(
                self.data.global_test_records
            ),
            "input_length": (
                self.data.input_length
            ),
            "num_classes": (
                self.data.num_classes
            ),
            "federated_rounds": (
                self.training.federated_rounds
            ),
            "local_epochs": (
                self.training.local_epochs
            ),
            "strategy": (
                self.strategy.name
            ),
            "proximal_mu": (
                self.strategy.proximal_mu
            ),
            "smotetomek": (
                self.strategy.smotetomek
            ),
            "ablation_seeds": list(
                self.ablation.seeds
            ),
            "num_ablation_definitions": len(
                self.ablation.experiments
            ),
        }


def main() -> None:
    config = ProjectConfig.load()

    print("Project configuration validated")

    for key, value in (
        config.summary().items()
    ):
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()