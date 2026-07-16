from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from flwr.simulation import run_simulation


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from fl_ecg_orchestrator.federated.client_app import (
    app as client_app,
)
from fl_ecg_orchestrator.federated.server_app import (
    app as server_app,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one federated ECG experiment."
        )
    )

    parser.add_argument(
        "--strategy",
        choices=["fedavg", "fedprox"],
        default="fedavg",
    )

    parser.add_argument(
        "--mu",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--smotetomek",
        action="store_true",
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--local-epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--clients",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--cpus-per-client",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    return parser.parse_args()


def set_runtime_environment(
    args: argparse.Namespace,
) -> None:
    values = {
        "FL_ECG_STRATEGY": args.strategy,
        "FL_ECG_PROXIMAL_MU": str(args.mu),
        "FL_ECG_SMOTETOMEK": str(
            args.smotetomek
        ).lower(),
        "FL_ECG_NUM_SERVER_ROUNDS": str(
            args.rounds
        ),
        "FL_ECG_LOCAL_EPOCHS": str(
            args.local_epochs
        ),
        "FL_ECG_LEARNING_RATE": str(
            args.learning_rate
        ),
        "FL_ECG_SEED": str(args.seed),
    }

    os.environ.update(values)


def validate_arguments(
    args: argparse.Namespace,
) -> None:
    if args.clients != 5:
        raise ValueError(
            "The current configuration contains "
            "exactly five clients."
        )

    if args.rounds <= 0:
        raise ValueError(
            "Rounds must be greater than zero."
        )

    if args.local_epochs <= 0:
        raise ValueError(
            "Local epochs must be greater than zero."
        )

    if (
        args.strategy == "fedprox"
        and args.mu <= 0
    ):
        raise ValueError(
            "FedProx requires mu greater than zero."
        )

    if (
        args.strategy == "fedavg"
        and args.mu != 0
    ):
        raise ValueError(
            "FedAvg must use mu equal to zero."
        )


def main() -> None:
    args = parse_arguments()

    validate_arguments(args)
    set_runtime_environment(args)

    print()
    print("Federated ECG experiment")
    print(f"Strategy: {args.strategy}")
    print(f"Mu: {args.mu}")
    print(
        f"SMOTETomek: "
        f"{args.smotetomek}"
    )
    print(f"Rounds: {args.rounds}")
    print(
        f"Local epochs: "
        f"{args.local_epochs}"
    )
    print(f"Seed: {args.seed}")
    print()

    backend_config = {
        "client_resources": {
            "num_cpus": (
                args.cpus_per_client
            ),
        }
    }

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=args.clients,
        backend_name="ray",
        backend_config=backend_config,
        verbose_logging=args.verbose,
    )


if __name__ == "__main__":
    main()