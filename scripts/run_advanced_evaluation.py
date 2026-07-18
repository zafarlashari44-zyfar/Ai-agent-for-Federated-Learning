from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from fl_ecg_orchestrator.data.loader import (
    load_config,
)
from fl_ecg_orchestrator.evaluation.global_evaluator import (
    GlobalEvaluator,
)


def find_latest_checkpoint(
    config_path: str,
) -> Path:

    config = load_config(
        config_path
    )

    checkpoint_directory = Path(
        config["project"]["output_dir"]
    ) / "checkpoints"

    if not checkpoint_directory.is_absolute():
        checkpoint_directory = (
            PROJECT_ROOT
            / checkpoint_directory
        )

    candidates = sorted(
        checkpoint_directory.glob(
            "*.pth"
        ),
        key=lambda path: (
            path.stat().st_mtime
        ),
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            "No checkpoint files were found "
            f"in {checkpoint_directory}"
        )

    return candidates[0]


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run research-grade evaluation "
            "for the global ECG model."
        )
    )

    parser.add_argument(
        "--config",
        default=(
            "fl_ecg_orchestrator/"
            "config/config.yaml"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
    )

    parser.add_argument(
        "--output",
        default="outputs/evaluation",
    )

    parser.add_argument(
        "--device",
        choices=[
            "cpu",
            "cuda",
        ],
        default=None,
    )

    return parser.parse_args()


def main() -> None:

    arguments = parse_arguments()

    checkpoint_path = (
        Path(
            arguments.checkpoint
        ).resolve()
        if arguments.checkpoint
        else find_latest_checkpoint(
            arguments.config
        )
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    print(
        "Running advanced global evaluation..."
    )

    evaluator = GlobalEvaluator(
        config_path=arguments.config,
        device=arguments.device,
    )

    results = evaluator.advanced_evaluation(
        checkpoint_path=checkpoint_path,
        output_dir=arguments.output,
    )

    metrics = results["metrics"]

    print()
    print(
        "Advanced evaluation complete."
    )
    print(
        f"Samples: {results['samples']}"
    )
    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )
    print(
        f"Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"Macro F1: "
        f"{metrics['macro_f1']:.4f}"
    )
    print(
        f"Weighted F1: "
        f"{metrics['weighted_f1']:.4f}"
    )
    print(
        f"Macro ROC AUC: "
        f"{metrics['macro_roc_auc']}"
    )
    print(
        f"Results: "
        f"{results['output_dir']}"
    )


if __name__ == "__main__":
    main()