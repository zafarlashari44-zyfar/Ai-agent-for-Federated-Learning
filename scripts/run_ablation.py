from __future__ import annotations
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fl_ecg_orchestrator.config.project_config import (
    ExperimentDefinition,
    ProjectConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SIMULATION_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "run_simulation.py"
)


@dataclass(frozen=True)
class ExperimentRun:
    run_index: int
    experiment_name: str
    strategy: str
    proximal_mu: float
    smotetomek: bool
    seed: int
    rounds: int
    local_epochs: int
    learning_rate: float

    @property
    def identifier(self) -> str:
        return (
            f"{self.run_index:03d}"
            f"_{self.strategy}"
            f"_mu_{self.proximal_mu}"
            f"_smote_{int(self.smotetomek)}"
            f"_seed_{self.seed}"
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the federated ECG ablation matrix "
            "defined in config.yaml."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=(
            "fl_ecg_orchestrator/config/config.yaml"
        ),
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--local-epochs",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
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
        "--only",
        choices=[
            "all",
            "fedavg",
            "fedavg_smotetomek",
            "fedprox",
            "fedprox_smotetomek",
        ],
        default="all",
    )

    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        dest="seeds",
    )

    parser.add_argument(
        "--mu",
        type=float,
        action="append",
        dest="mu_values",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    return parser.parse_args()


def validate_arguments(
    args: argparse.Namespace,
) -> None:
    if args.rounds is not None and args.rounds <= 0:
        raise ValueError(
            "Rounds must be greater than zero."
        )

    if (
        args.local_epochs is not None
        and args.local_epochs <= 0
    ):
        raise ValueError(
            "Local epochs must be greater than zero."
        )

    if (
        args.learning_rate is not None
        and args.learning_rate <= 0
    ):
        raise ValueError(
            "Learning rate must be greater than zero."
        )

    if args.clients <= 0:
        raise ValueError(
            "Clients must be greater than zero."
        )

    if args.cpus_per_client <= 0:
        raise ValueError(
            "CPU allocation must be greater than zero."
        )

    if args.mu_values:
        invalid_mu_values = [
            value
            for value in args.mu_values
            if value <= 0
        ]

        if invalid_mu_values:
            raise ValueError(
                "Every FedProx mu value must be "
                "greater than zero."
            )


def experiment_matches_filter(
    experiment: ExperimentDefinition,
    selected_name: str,
) -> bool:
    if selected_name == "all":
        return True

    return experiment.name == selected_name


def resolve_mu_values(
    experiment: ExperimentDefinition,
    requested_mu_values: list[float] | None,
) -> tuple[float, ...]:
    if experiment.strategy == "fedavg":
        return (0.0,)

    if requested_mu_values:
        return tuple(
            float(value)
            for value in requested_mu_values
        )

    if experiment.proximal_mu_values:
        return experiment.proximal_mu_values

    if experiment.proximal_mu is not None:
        return (
            float(experiment.proximal_mu),
        )

    raise ValueError(
        f"FedProx experiment {experiment.name} "
        "does not define a proximal mu value."
    )


def build_experiment_matrix(
    config: ProjectConfig,
    args: argparse.Namespace,
) -> list[ExperimentRun]:
    seeds = tuple(
        args.seeds
        if args.seeds
        else config.ablation.seeds
    )

    if not seeds:
        seeds = (
            config.project.seed,
        )

    rounds = (
        args.rounds
        if args.rounds is not None
        else config.training.federated_rounds
    )

    local_epochs = (
        args.local_epochs
        if args.local_epochs is not None
        else config.training.local_epochs
    )

    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else config.training.learning_rate
    )

    if args.quick:
        seeds = (
            seeds[0],
        )
        rounds = 1
        local_epochs = 1

    runs: list[ExperimentRun] = []

    for experiment in config.ablation.experiments:
        if not experiment_matches_filter(
            experiment,
            args.only,
        ):
            continue

        mu_values = resolve_mu_values(
            experiment=experiment,
            requested_mu_values=(
                args.mu_values
            ),
        )

        if args.quick:
            mu_values = (
                mu_values[0],
            )

        for seed in seeds:
            for proximal_mu in mu_values:
                runs.append(
                    ExperimentRun(
                        run_index=len(runs) + 1,
                        experiment_name=(
                            experiment.name
                        ),
                        strategy=(
                            experiment.strategy
                        ),
                        proximal_mu=float(
                            proximal_mu
                        ),
                        smotetomek=(
                            experiment.smotetomek
                        ),
                        seed=int(seed),
                        rounds=int(rounds),
                        local_epochs=int(
                            local_epochs
                        ),
                        learning_rate=float(
                            learning_rate
                        ),
                    )
                )

    if not runs:
        raise ValueError(
            "The selected filters produced no "
            "ablation experiments."
        )

    return runs


def create_run_directory(
    output_root: Path,
) -> Path:
    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    run_directory = (
        output_root
        / "ablation"
        / f"ablation_{timestamp}"
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    (
        run_directory
        / "logs"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    return run_directory


def build_command(
    experiment: ExperimentRun,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(SIMULATION_SCRIPT),
        "--strategy",
        experiment.strategy,
        "--mu",
        str(
            experiment.proximal_mu
        ),
        "--rounds",
        str(
            experiment.rounds
        ),
        "--local-epochs",
        str(
            experiment.local_epochs
        ),
        "--learning-rate",
        str(
            experiment.learning_rate
        ),
        "--seed",
        str(
            experiment.seed
        ),
        "--clients",
        str(
            args.clients
        ),
        "--cpus-per-client",
        str(
            args.cpus_per_client
        ),
    ]

    if experiment.smotetomek:
        command.append(
            "--smotetomek"
        )

    if args.verbose:
        command.append(
            "--verbose"
        )

    return command


def write_json(
    path: Path,
    payload: Any,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
        )


def find_latest_summary(
    metrics_root: Path,
    experiment: ExperimentRun,
    started_at: float,
) -> dict[str, Any] | None:
    if not metrics_root.exists():
        return None

    matching_summaries: list[Path] = []

    expected_name = (
        f"{experiment.strategy}"
        f"_mu_{experiment.proximal_mu}"
        f"_smote_{int(experiment.smotetomek)}"
        f"_seed_{experiment.seed}"
    )

    for summary_path in metrics_root.rglob(
        "summary.json"
    ):
        if (
            summary_path.stat().st_mtime
            < started_at
        ):
            continue

        try:
            with summary_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                payload = json.load(
                    file
                )
        except (
            OSError,
            json.JSONDecodeError,
        ):
            continue

        if (
            payload.get(
                "experiment_name"
            )
            == expected_name
        ):
            matching_summaries.append(
                summary_path
            )

    if not matching_summaries:
        return None

    latest_path = max(
        matching_summaries,
        key=lambda path: (
            path.stat().st_mtime
        ),
    )

    with latest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(
            file
        )

    payload["_summary_path"] = str(
        latest_path
    )

    return payload


def extract_result_row(
    experiment: ExperimentRun,
    status: str,
    duration_seconds: float,
    return_code: int,
    summary: dict[str, Any] | None,
    log_path: Path,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_index": experiment.run_index,
        "identifier": experiment.identifier,
        "experiment_name": (
            experiment.experiment_name
        ),
        "strategy": experiment.strategy,
        "proximal_mu": (
            experiment.proximal_mu
        ),
        "smotetomek": (
            experiment.smotetomek
        ),
        "seed": experiment.seed,
        "rounds": experiment.rounds,
        "local_epochs": (
            experiment.local_epochs
        ),
        "learning_rate": (
            experiment.learning_rate
        ),
        "status": status,
        "return_code": return_code,
        "duration_seconds": round(
            duration_seconds,
            3,
        ),
        "log_path": str(
            log_path
        ),
        "summary_path": None,
        "best_round": None,
        "best_macro_f1": None,
        "accuracy": None,
        "macro_precision": None,
        "macro_recall": None,
        "macro_f1": None,
        "recall_N": None,
        "recall_S": None,
        "recall_V": None,
        "recall_F": None,
        "recall_Q": None,
        "f1_N": None,
        "f1_S": None,
        "f1_V": None,
        "f1_F": None,
        "f1_Q": None,
        "final_checkpoint_path": None,
        "best_checkpoint_path": None,
    }

    if summary is None:
        return row

    global_metrics = summary.get(
        "global_test_metrics",
        {},
    )

    best_model = summary.get(
        "best_model",
        {},
    )

    recall = global_metrics.get(
        "per_class_recall",
        {},
    )

    f1 = global_metrics.get(
        "per_class_f1",
        {},
    )

    row.update(
        {
            "summary_path": summary.get(
                "_summary_path"
            ),
            "best_round": best_model.get(
                "best_round"
            ),
            "best_macro_f1": (
                best_model.get(
                    "best_macro_f1"
                )
            ),
            "accuracy": global_metrics.get(
                "accuracy"
            ),
            "macro_precision": (
                global_metrics.get(
                    "macro_precision"
                )
            ),
            "macro_recall": (
                global_metrics.get(
                    "macro_recall"
                )
            ),
            "macro_f1": global_metrics.get(
                "macro_f1"
            ),
            "recall_N": recall.get(
                "N"
            ),
            "recall_S": recall.get(
                "S"
            ),
            "recall_V": recall.get(
                "V"
            ),
            "recall_F": recall.get(
                "F"
            ),
            "recall_Q": recall.get(
                "Q"
            ),
            "f1_N": f1.get(
                "N"
            ),
            "f1_S": f1.get(
                "S"
            ),
            "f1_V": f1.get(
                "V"
            ),
            "f1_F": f1.get(
                "F"
            ),
            "f1_Q": f1.get(
                "Q"
            ),
            "final_checkpoint_path": (
                summary.get(
                    "final_checkpoint_path"
                )
            ),
            "best_checkpoint_path": (
                best_model.get(
                    "best_checkpoint_path"
                )
            ),
        }
    )

    return row


def write_results_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def run_single_experiment(
    experiment: ExperimentRun,
    args: argparse.Namespace,
    run_directory: Path,
    metrics_root: Path,
) -> dict[str, Any]:
    command = build_command(
        experiment=experiment,
        args=args,
    )

    log_path = (
        run_directory
        / "logs"
        / f"{experiment.identifier}.log"
    )

    print()
    print(
        f"Starting run "
        f"{experiment.run_index}"
    )
    print(
        f"Experiment "
        f"{experiment.experiment_name}"
    )
    print(
        f"Strategy "
        f"{experiment.strategy}"
    )
    print(
        f"Mu "
        f"{experiment.proximal_mu}"
    )
    print(
        f"SMOTETomek "
        f"{experiment.smotetomek}"
    )
    print(
        f"Seed "
        f"{experiment.seed}"
    )
    print(
        f"Log "
        f"{log_path}"
    )
    print()

    started_at = time.time()

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if process.stdout is None:
            raise RuntimeError(
                "Unable to read experiment output."
            )

        for line in process.stdout:
            print(
                line,
                end="",
            )
            log_file.write(
                line
            )
            log_file.flush()

        return_code = process.wait()

    duration_seconds = (
        time.time()
        - started_at
    )

    summary = find_latest_summary(
        metrics_root=metrics_root,
        experiment=experiment,
        started_at=started_at,
    )

    status = (
        "completed"
        if return_code == 0
        else "failed"
    )

    return extract_result_row(
        experiment=experiment,
        status=status,
        duration_seconds=(
            duration_seconds
        ),
        return_code=return_code,
        summary=summary,
        log_path=log_path,
    )


def print_plan(
    runs: list[ExperimentRun],
) -> None:
    print()
    print(
        f"Planned experiments "
        f"{len(runs)}"
    )

    for experiment in runs:
        print(
            f"{experiment.run_index:03d} "
            f"{experiment.experiment_name} "
            f"strategy={experiment.strategy} "
            f"mu={experiment.proximal_mu} "
            f"smote={experiment.smotetomek} "
            f"seed={experiment.seed} "
            f"rounds={experiment.rounds} "
            f"epochs={experiment.local_epochs}"
        )


def main() -> None:
    args = parse_arguments()

    validate_arguments(
        args
    )

    config = ProjectConfig.load(
        args.config
    )

    runs = build_experiment_matrix(
        config=config,
        args=args,
    )

    print_plan(
        runs
    )

    if args.dry_run:
        print()
        print(
            "Dry run completed"
        )
        return

    run_directory = create_run_directory(
        config.project.output_dir
    )

    metrics_root = (
        config.project.output_dir
        / "metrics"
    )

    manifest_path = (
        run_directory
        / "experiment_manifest.json"
    )

    results_path = (
        run_directory
        / "ablation_results.csv"
    )

    status_path = (
        run_directory
        / "run_status.json"
    )

    write_json(
        manifest_path,
        {
            "created_at": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "project": (
                config.summary()
            ),
            "arguments": vars(
                args
            ),
            "experiments": [
                asdict(
                    experiment
                )
                for experiment in runs
            ],
        },
    )

    results: list[
        dict[str, Any]
    ] = []

    for experiment in runs:
        result = run_single_experiment(
            experiment=experiment,
            args=args,
            run_directory=run_directory,
            metrics_root=metrics_root,
        )

        results.append(
            result
        )

        write_results_csv(
            results_path,
            results,
        )

        write_json(
            status_path,
            {
                "completed_runs": len(
                    results
                ),
                "total_runs": len(
                    runs
                ),
                "latest_result": (
                    result
                ),
                "results": results,
            },
        )

        if (
            result["status"]
            == "failed"
            and not args.continue_on_error
        ):
            raise RuntimeError(
                f"Experiment failed. "
                f"See {result['log_path']}"
            )

    completed_count = sum(
        result["status"]
        == "completed"
        for result in results
    )

    failed_count = sum(
        result["status"]
        == "failed"
        for result in results
    )

    print()
    print(
        "Ablation run completed"
    )
    print(
        f"Completed "
        f"{completed_count}"
    )
    print(
        f"Failed "
        f"{failed_count}"
    )
    print(
        f"Results "
        f"{results_path}"
    )
    print(
        f"Manifest "
        f"{manifest_path}"
    )


if __name__ == "__main__":
    main()