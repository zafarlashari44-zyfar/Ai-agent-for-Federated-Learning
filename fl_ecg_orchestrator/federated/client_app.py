from __future__ import annotations

from typing import Any

from flwr.app import (
    ArrayRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.clientapp import ClientApp

from fl_ecg_orchestrator.config.project_config import (
    ProjectConfig,
)
from fl_ecg_orchestrator.config.runtime import (
    get_runtime_value,
)
from fl_ecg_orchestrator.federated.local_trainer import (
    LocalTrainer,
)


CONFIG_PATH = (
    "fl_ecg_orchestrator/config/config.yaml"
)

app = ClientApp()


def resolve_client_id(
    context: Context,
) -> str:
    partition_id = int(
        context.node_config["partition-id"]
    )

    config = ProjectConfig.load(
        CONFIG_PATH
    )

    client_ids = config.client_ids()

    if not 0 <= partition_id < len(client_ids):
        raise ValueError(
            f"Invalid Flower partition ID "
            f"{partition_id}"
        )

    return client_ids[partition_id]


def create_local_trainer(
    context: Context,
    client_id: str,
) -> LocalTrainer:
    config = ProjectConfig.load(
        CONFIG_PATH
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

    proximal_mu = float(
        get_runtime_value(
            context=context,
            name="proximal-mu",
            default=(
                config.strategy.proximal_mu
            ),
        )
    )

    seed = int(
        get_runtime_value(
            context=context,
            name="seed",
            default=config.project.seed,
        )
    )

    return LocalTrainer(
        client_id=client_id,
        config=config,
        use_smotetomek=use_smotetomek,
        proximal_mu=proximal_mu,
        seed=seed,
    )


def build_training_metrics(
    client_index: int,
    result: dict[str, Any],
    use_smotetomek: bool,
    proximal_mu: float,
) -> MetricRecord:
    final_validation = result[
        "final_validation"
    ]

    final_epoch = result[
        "epoch_history"
    ][-1]

    return MetricRecord(
        {
            "client-index": int(
                client_index
            ),
            "num-examples": int(
                result["num_examples"]
            ),
            "training-loss": float(
                final_epoch["training_loss"]
            ),
            "training-accuracy": float(
                final_epoch[
                    "training_accuracy"
                ]
            ),
            "validation-loss": float(
                final_validation["loss"]
            ),
            "validation-accuracy": float(
                final_validation["accuracy"]
            ),
            "validation-macro-f1": float(
                final_validation["macro_f1"]
            ),
            "recall-N": float(
                final_validation[
                    "per_class_recall"
                ]["N"]
            ),
            "recall-S": float(
                final_validation[
                    "per_class_recall"
                ]["S"]
            ),
            "recall-V": float(
                final_validation[
                    "per_class_recall"
                ]["V"]
            ),
            "recall-F": float(
                final_validation[
                    "per_class_recall"
                ]["F"]
            ),
            "recall-Q": float(
                final_validation[
                    "per_class_recall"
                ]["Q"]
            ),
            "f1-N": float(
                final_validation[
                    "per_class_f1"
                ]["N"]
            ),
            "f1-S": float(
                final_validation[
                    "per_class_f1"
                ]["S"]
            ),
            "f1-V": float(
                final_validation[
                    "per_class_f1"
                ]["V"]
            ),
            "f1-F": float(
                final_validation[
                    "per_class_f1"
                ]["F"]
            ),
            "f1-Q": float(
                final_validation[
                    "per_class_f1"
                ]["Q"]
            ),
            "duration-seconds": float(
                result["duration_seconds"]
            ),
            "proximal-mu": float(
                proximal_mu
            ),
            "smotetomek": int(
                use_smotetomek
            ),
        }
    )


def build_evaluation_metrics(
    client_index: int,
    result: dict[str, Any],
) -> MetricRecord:
    return MetricRecord(
        {
            "client-index": int(
                client_index
            ),
            "num-examples": int(
                result["num_samples"]
            ),
            "loss": float(
                result["loss"]
            ),
            "accuracy": float(
                result["accuracy"]
            ),
            "macro-precision": float(
                result["macro_precision"]
            ),
            "macro-recall": float(
                result["macro_recall"]
            ),
            "macro-f1": float(
                result["macro_f1"]
            ),
            "recall-N": float(
                result[
                    "per_class_recall"
                ]["N"]
            ),
            "recall-S": float(
                result[
                    "per_class_recall"
                ]["S"]
            ),
            "recall-V": float(
                result[
                    "per_class_recall"
                ]["V"]
            ),
            "recall-F": float(
                result[
                    "per_class_recall"
                ]["F"]
            ),
            "recall-Q": float(
                result[
                    "per_class_recall"
                ]["Q"]
            ),
            "f1-N": float(
                result[
                    "per_class_f1"
                ]["N"]
            ),
            "f1-S": float(
                result[
                    "per_class_f1"
                ]["S"]
            ),
            "f1-V": float(
                result[
                    "per_class_f1"
                ]["V"]
            ),
            "f1-F": float(
                result[
                    "per_class_f1"
                ]["F"]
            ),
            "f1-Q": float(
                result[
                    "per_class_f1"
                ]["Q"]
            ),
        }
    )


@app.train()
def train(
    message: Message,
    context: Context,
) -> Message:
    client_id = resolve_client_id(
        context
    )

    client_index = int(
        context.node_config["partition-id"]
    )

    trainer = create_local_trainer(
        context=context,
        client_id=client_id,
    )

    received_state = (
        message.content["arrays"]
        .to_torch_state_dict()
    )

    trainer.model.load_state_dict(
        received_state,
        strict=True,
    )

    message_config = message.content[
        "config"
    ]

    local_epochs = int(
        message_config.get(
            "local-epochs",
            get_runtime_value(
                context=context,
                name="local-epochs",
                default=(
                    trainer.config
                    .training
                    .local_epochs
                ),
            ),
        )
    )

    learning_rate = float(
        message_config.get(
            "learning-rate",
            get_runtime_value(
                context=context,
                name="learning-rate",
                default=(
                    trainer.config
                    .training
                    .learning_rate
                ),
            ),
        )
    )

    proximal_mu = float(
        message_config.get(
            "proximal-mu",
            get_runtime_value(
                context=context,
                name="proximal-mu",
                default=(
                    trainer.proximal_mu
                ),
            ),
        )
    )

    result = trainer.train(
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        proximal_mu=proximal_mu,
    )

    model_record = ArrayRecord(
        trainer.model.state_dict()
    )

    metrics_record = build_training_metrics(
        client_index=client_index,
        result=result,
        use_smotetomek=(
            trainer.use_smotetomek
        ),
        proximal_mu=proximal_mu,
    )

    content = RecordDict(
        {
            "arrays": model_record,
            "metrics": metrics_record,
        }
    )

    return Message(
        content=content,
        reply_to=message,
    )


@app.evaluate()
def evaluate(
    message: Message,
    context: Context,
) -> Message:
    client_id = resolve_client_id(
        context
    )

    client_index = int(
        context.node_config["partition-id"]
    )

    trainer = create_local_trainer(
        context=context,
        client_id=client_id,
    )

    received_state = (
        message.content["arrays"]
        .to_torch_state_dict()
    )

    trainer.model.load_state_dict(
        received_state,
        strict=True,
    )

    result = trainer.evaluate()

    metrics_record = (
        build_evaluation_metrics(
            client_index=client_index,
            result=result,
        )
    )

    content = RecordDict(
        {
            "metrics": metrics_record,
        }
    )

    return Message(
        content=content,
        reply_to=message,
    )