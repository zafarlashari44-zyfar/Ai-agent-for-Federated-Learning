from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fl_ecg_orchestrator.data.loader import load_config, resolve_project_path
from fl_ecg_orchestrator.explainability.report_builder import build_reports
from fl_ecg_orchestrator.explainability.shap_explainer import ECGSHAPExplainer
from fl_ecg_orchestrator.explainability.shap_plots import (
    save_class_importance_plots,
    save_global_feature_importance,
    save_individual_explanation,
    save_summary_plot,
)


def find_latest_checkpoint(config_path: str) -> Path:
    config = load_config(config_path)
    checkpoint_dir = (
        resolve_project_path(config["project"]["output_dir"])
        / "checkpoints"
    )

    candidates = sorted(
        checkpoint_dir.glob("*.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No .pth checkpoints found in {checkpoint_dir}. "
            "Pass --checkpoint explicitly or run training first."
        )

    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SHAP explanations for the global ECG test set."
    )
    parser.add_argument(
        "--config",
        default="fl_ecg_orchestrator/config/config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. Defaults to the newest .pth checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="outputs/explainability/shap",
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--explain-size",
        type=int,
        default=250,
    )
    parser.add_argument(
        "--individual-plots",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint = (
        resolve_project_path(args.checkpoint)
        if args.checkpoint
        else find_latest_checkpoint(args.config)
    )

    print(f"Checkpoint: {checkpoint}")
    print("Loading model and global test data...")

    explainer = ECGSHAPExplainer(
        checkpoint_path=checkpoint,
        config_path=args.config,
        background_size=args.background_size,
        explain_size=args.explain_size,
        device=args.device,
    )

    print(
        f"Computing SHAP values with {explainer.explainer_name} "
        f"on {explainer.device}..."
    )

    results = explainer.run()
    output_dir = explainer.save_raw_results(args.output, results)

    save_global_feature_importance(results, output_dir)
    save_summary_plot(results, output_dir)
    save_class_importance_plots(results, output_dir)
    build_reports(results, output_dir)

    plot_count = min(
        max(args.individual_plots, 0),
        len(results["features"]),
    )

    for position in range(plot_count):
        save_individual_explanation(
            results=results,
            sample_position=position,
            output_dir=output_dir / "individual",
        )

    accuracy = (
        results["predictions"] == results["true_labels"]
    ).mean()

    print()
    print("SHAP analysis complete.")
    print(f"Explainer: {results['explainer']}")
    print(f"Explained samples: {len(results['features'])}")
    print(f"Subset accuracy: {accuracy:.4f}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()

