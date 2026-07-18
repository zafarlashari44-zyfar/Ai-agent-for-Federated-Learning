from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fl_ecg_orchestrator.evaluation.calibration import CalibrationEvaluator


def find_latest_checkpoint() -> Path:
    candidates = [
        PROJECT_ROOT / "fl_ecg_orchestrator" / "outputs" / "checkpoints",
        PROJECT_ROOT / "outputs" / "checkpoints",
    ]

    checkpoint_files: list[Path] = []

    for directory in candidates:
        if directory.exists():
            checkpoint_files.extend(directory.glob("*.pth"))

    if not checkpoint_files:
        raise FileNotFoundError(
            "No checkpoint found. Pass one explicitly with --checkpoint."
        )

    final_checkpoints = [
        path
        for path in checkpoint_files
        if "final" in path.name.lower()
    ]

    selected_pool = final_checkpoints or checkpoint_files

    return max(
        selected_pool,
        key=lambda path: path.stat().st_mtime,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ECG model calibration evaluation."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to a trained .pth checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Evaluation device. Defaults to CUDA when available.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=15,
        help="Number of calibration bins.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "calibration",
        help="Directory for calibration outputs.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(
            PROJECT_ROOT
            / "fl_ecg_orchestrator"
            / "config"
            / "config.yaml"
        ),
        help="Path to config.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else find_latest_checkpoint().resolve()
    )

    print(f"Checkpoint: {checkpoint}")
    print("Running calibration evaluation...")

    evaluator = CalibrationEvaluator(
        config_path=args.config,
        device=args.device,
        num_bins=args.bins,
    )

    result = evaluator.run(
        checkpoint_path=checkpoint,
        output_dir=args.output_dir,
    )

    metrics = result["metrics"]

    print()
    print("Calibration evaluation complete.")
    print(f"Samples: {metrics['samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(
        "Average confidence: "
        f"{metrics['average_confidence']:.4f}"
    )
    print(
        "Expected Calibration Error: "
        f"{metrics['expected_calibration_error']:.4f}"
    )
    print(
        "Maximum Calibration Error: "
        f"{metrics['maximum_calibration_error']:.4f}"
    )
    print(
        "Multiclass Brier score: "
        f"{metrics['multiclass_brier_score']:.4f}"
    )
    print(f"Results: {result['output_dir']}")


if __name__ == "__main__":
    main()
