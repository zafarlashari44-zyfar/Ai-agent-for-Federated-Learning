from flwr.serverapp.strategy import FedAvg, FedProx


def create_strategy(
    strategy_name: str,
    num_clients: int,
    proximal_mu: float,
):
    name = strategy_name.strip().lower()

    shared_arguments = {
        "fraction_train": 1.0,
        "fraction_evaluate": 1.0,
        "min_train_nodes": num_clients,
        "min_evaluate_nodes": num_clients,
        "min_available_nodes": num_clients,
    }

    if name == "fedavg":
        return FedAvg(**shared_arguments)

    if name == "fedprox":
        if proximal_mu <= 0:
            raise ValueError(
                "FedProx requires proximal_mu greater than zero"
            )

        return FedProx(
            **shared_arguments,
            proximal_mu=proximal_mu,
        )

    raise ValueError(
        f"Unsupported federated strategy {strategy_name}"
    )
