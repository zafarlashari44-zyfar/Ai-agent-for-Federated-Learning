from __future__ import annotations

from typing import Any

import torch
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    MetricRecord,
    RecordDict,
)
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedProx

from fl_ecg_orchestrator.config.project_config import (
    ProjectConfig,
)
from fl_ecg_orchestrator.config.runtime import (
    get_runtime_value,
)
from fl_ecg_orchestrator.evaluation.central_evaluator import (
    CentralEvaluator,
)
from fl_ecg_orchestrator.evaluation.experiment_logger import (
    ExperimentLogger,
)
from fl_ecg_orchestrator.evaluation.global_evaluator import (
    GlobalEvaluator,
)
from fl_ecg_orchestrator.model.cnn1d import (
    create_model,
)


CONFIG_PATH = (
    "fl_ecg_orchestrator/config/config.yaml"
)

app = ServerApp()


def weighted_metric_average(
    records: list[RecordDict],
    metric_name: str,
    weighted_by_key: str,
) -> float:
    weighted_sum = 0.0
    total_weight = 0.0

    for record in records:
        metrics = record["metrics"]

        weight = float(
            metrics.get(
                weighted_by_key,
                0,
            )
        )

        value = metrics.get(
            metric_name
        )

        if value is None:
            continue

        weighted_sum += (
            float(value) * weight
        )

        total_weight += weight

    return weighted_sum / max(
        total_weight,
        1.0,
    )


def aggregate_metrics(
    records: list[RecordDict],
    weighted_by_key: str,
    metric_names: list[str],
) -> MetricRecord:
    total_examples = sum(
        int(
            record["metrics"].get(
                weighted_by_key,
                0,
            )
        )
        for record in records
    )

    aggregated: dict[str, int | float] = {
        weighted_by_key: int(
            total_examples
        )
    }

    for metric_name in metric_names:
        aggregated[metric_name] = (
            weighted_metric_average(
                records=records,
                metric_name=metric_name,
                weighted_by_key=(
                    weighted_by_key
                ),
            )
        )

    return MetricRecord(
        aggregated
    )


def aggregate_client_training_metrics(
    records: list[RecordDict],
    weighted_by_key: str,
) -> MetricRecord:
    metric_names = [
        "training-loss",
        "training-accuracy",
        "validation-loss",
        "validation-accuracy",
        "validation-macro-f1",
        "recall-N",
        "recall-S",
        "recall-V",
        "recall-F",
        "recall-Q",
        "f1-N",
        "f1-S",
        "f1-V",
        "f1-F",
        "f1-Q",
        "duration-seconds",
    ]

    return aggregate_metrics(
        records=records,
        weighted_by_key=(
            weighted_by_key
        ),
        metric_names=metric_names,
    )


def aggregate_client_evaluation_metrics(
    records: list[RecordDict],
    weighted_by_key: str,
) -> MetricRecord:
    metric_names = [
        "loss",
        "accuracy",
        "macro-precision",
        "macro-recall",
        "macro-f1",
        "recall-N",
        "recall-S",
        "recall-V",
        "recall-F",
        "recall-Q",
        "f1-N",
        "f1-S",
        "f1-V",
        "f1-F",
        "f1-Q",
    ]

    return aggregate_metrics(
        records=records,
        weighted_by_key=(
            weighted_by_key
        ),
        metric_names=metric_names,
    )


def create_federated_strategy(
    strategy_name: str,
    proximal_mu: float,
    fraction_evaluate: float,
    num_clients: int,
) -> FedAvg | FedProx:
    normalized_name = (
        strategy_name
        .strip()
        .lower()
    )

    shared_arguments = {
        "fraction_train": 1.0,
        "fraction_evaluate": (
            fraction_evaluate
        ),
        "min_train_nodes": (
            num_clients
        ),
        "min_evaluate_nodes": (
            num_clients
        ),
        "min_available_nodes": (
            num_clients
        ),
        "weighted_by_key": (
            "num-examples"
        ),
        "train_metrics_aggr_fn": (
            aggregate_client_training_metrics
        ),
        "evaluate_metrics_aggr_fn": (
            aggregate_client_evaluation_metrics
        ),
    }

    if normalized_name == "fedavg":
        return FedAvg(
            **shared_arguments
        )

    if normalized_name == "fedprox":
        if proximal_mu <= 0:
            raise ValueError(
                "FedProx requires proximal_mu "
                "greater than zero"
            )

        return FedProx(
            **shared_arguments,
            proximal_mu=proximal_mu,
        )

    raise ValueError(
        f"Unsupported federated strategy "
        f"{strategy_name}"
    )


def serialise_metrics(
    value: Any,
) -> Any:
    if isinstance(
        value,
        dict,
    ):
        return {
            key: serialise_metrics(item)
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        list,
    ):
        return [
            serialise_metrics(item)
            for item in value
        ]

    if isinstance(
        value,
        tuple,
    ):
        return [
            serialise_metrics(item)
            for item in value
        ]

    if isinstance(
        value,
        torch.Tensor,
    ):
        return (
            value
            .detach()
            .cpu()
            .tolist()
        )

    if hasattr(
        value,
        "item",
    ):
        return value.item()

    return value


@app.main()
def main(
    grid: Grid,
    context: Context,
) -> None:
    config = ProjectConfig.load(
        CONFIG_PATH
    )

    strategy_name = str(
        get_runtime_value(
            context=context,
            name="strategy",
            default=(
                config.strategy.name
            ),
        )
    ).lower()

    proximal_mu = float(
        get_runtime_value(
            context=context,
            name="proximal-mu",
            default=(
                config.strategy.proximal_mu
            ),
        )
    )

    num_rounds = int(
        get_runtime_value(
            context=context,
            name="num-server-rounds",
            default=(
                config.training
                .federated_rounds
            ),
        )
    )

    local_epochs = int(
        get_runtime_value(
            context=context,
            name="local-epochs",
            default=(
                config.training
                .local_epochs
            ),
        )
    )

    learning_rate = float(
        get_runtime_value(
            context=context,
            name="learning-rate",
            default=(
                config.training
                .learning_rate
            ),
        )
    )

    fraction_evaluate = float(
        get_runtime_value(
            context=context,
            name="fraction-evaluate",
            default=1.0,
        )
    )

    use_smotetomek = bool(
        get_runtime_value(
            context=context,
            name="smotetomek",
            default=(
                config.strategy.smotetomek
            ),
        )
    )

    seed = int(
        get_runtime_value(
            context=context,
            name="seed",
            default=(
                config.project.seed
            ),
        )
    )

    save_model = bool(
        get_runtime_value(
            context=context,
            name="save-model",
            default=True,
        )
    )

    num_clients = len(
        config.clients
    )

    if strategy_name == "fedavg":
        proximal_mu = 0.0

    experiment_name = (
        f"{strategy_name}"
        f"_mu_{proximal_mu}"
        f"_smote_{int(use_smotetomek)}"
        f"_seed_{seed}"
    )

    print()
    print(
        "Starting federated experiment"
    )
    print(
        f"Experiment {experiment_name}"
    )
    print(
        f"Strategy {strategy_name}"
    )
    print(
        f"Rounds {num_rounds}"
    )
    print(
        f"Local epochs {local_epochs}"
    )
    print(
        f"Learning rate {learning_rate}"
    )
    print(
        f"Proximal mu {proximal_mu}"
    )
    print(
        f"SMOTETomek {use_smotetomek}"
    )
    print(
        f"Clients {num_clients}"
    )
    print()

    logger = ExperimentLogger(
        experiment_name=(
            experiment_name
        ),
        output_root=str(
            config.project.output_dir
        ),
    )

    global_evaluator = GlobalEvaluator(
        config_path=(
            config.source_path
        )
    )

    central_evaluator = (
        CentralEvaluator(
            evaluator=(
                global_evaluator
            ),
            logger=logger,
            experiment_name=(
                experiment_name
            ),
        )
    )

    global_model = create_model(
        input_length=(
            config.data.input_length
        ),
        num_classes=(
            config.data.num_classes
        ),
    )

    initial_arrays = ArrayRecord(
        global_model.state_dict()
    )

    strategy = (
        create_federated_strategy(
            strategy_name=(
                strategy_name
            ),
            proximal_mu=(
                proximal_mu
            ),
            fraction_evaluate=(
                fraction_evaluate
            ),
            num_clients=(
                num_clients
            ),
        )
    )

    logger.write_metadata(
        {
            "experiment_name": (
                experiment_name
            ),
            "project_name": (
                config.project.name
            ),
            "config_path": str(
                config.source_path
            ),
            "strategy": (
                strategy_name
            ),
            "proximal_mu": (
                proximal_mu
            ),
            "smotetomek": (
                use_smotetomek
            ),
            "seed": seed,
            "num_rounds": (
                num_rounds
            ),
            "local_epochs": (
                local_epochs
            ),
            "learning_rate": (
                learning_rate
            ),
            "fraction_evaluate": (
                fraction_evaluate
            ),
            "num_clients": (
                num_clients
            ),
            "client_ids": list(
                config.client_ids()
            ),
            "global_test_records": list(
                config.data
                .global_test_records
            ),
            "input_length": (
                config.data.input_length
            ),
            "num_classes": (
                config.data.num_classes
            ),
        }
    )

    train_config = ConfigRecord(
        {
            "local-epochs": (
                local_epochs
            ),
            "learning-rate": (
                learning_rate
            ),
            "proximal-mu": (
                proximal_mu
            ),
        }
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=(
            initial_arrays
        ),
        num_rounds=(
            num_rounds
        ),
        train_config=(
            train_config
        ),
        evaluate_fn=(
            central_evaluator
        ),
    )

    if result.arrays is None:
        raise RuntimeError(
            "Flower did not return "
            "final global model arrays"
        )

    final_state_dict = (
        result.arrays
        .to_torch_state_dict()
    )

    final_global_metrics = (
        global_evaluator
        .evaluate_state_dict(
            final_state_dict
        )
    )

    final_checkpoint_path = None

    if save_model:
        final_checkpoint_path = (
            global_evaluator
            .save_checkpoint(
                state_dict=(
                    final_state_dict
                ),
                experiment_name=(
                    f"{experiment_name}_final"
                ),
                round_number=(
                    num_rounds
                ),
                metrics=(
                    final_global_metrics
                ),
            )
        )

    summary = {
        "experiment_name": (
            experiment_name
        ),
        "project_name": (
            config.project.name
        ),
        "strategy": (
            strategy_name
        ),
        "proximal_mu": (
            proximal_mu
        ),
        "smotetomek": (
            use_smotetomek
        ),
        "seed": seed,
        "rounds": (
            num_rounds
        ),
        "local_epochs": (
            local_epochs
        ),
        "learning_rate": (
            learning_rate
        ),
        "final_checkpoint_path": (
            str(final_checkpoint_path)
            if final_checkpoint_path
            else None
        ),
        "best_model": (
            central_evaluator
            .get_summary()
        ),
        "global_test_metrics": (
            serialise_metrics(
                final_global_metrics
            )
        ),
    }

    logger.write_summary(
        summary
    )

    print()
    print(
        "Federated experiment completed"
    )
    print(
        f"Experiment {experiment_name}"
    )
    print(
        f"Final accuracy "
        f"{final_global_metrics['accuracy']:.4f}"
    )
    print(
        f"Final macro F1 "
        f"{final_global_metrics['macro_f1']:.4f}"
    )
    print(
        f"Best round "
        f"{central_evaluator.best_round}"
    )
    print(
        f"Best macro F1 "
        f"{central_evaluator.best_macro_f1:.4f}"
    )

    if (
        central_evaluator
        .best_checkpoint_path
        is not None
    ):
        print(
            f"Best checkpoint "
            f"{central_evaluator.best_checkpoint_path}"
        )

    if final_checkpoint_path is not None:
        print(
            f"Final checkpoint "
            f"{final_checkpoint_path}"
        )