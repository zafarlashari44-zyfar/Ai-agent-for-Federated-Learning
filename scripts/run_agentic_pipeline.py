from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fl_ecg_orchestrator.agents import PlannerAgent, PlannerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the master ECG AI agent and its specialized sub-agents."
        )
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mc-samples", type=int, default=30)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
    )

    parser.add_argument("--run-scribe", action="store_true")
    parser.add_argument(
        "--scribe-path",
        default="agent1_scribe.py",
    )
    parser.add_argument("--record-name", default=None)
    parser.add_argument("--clinical-text", default=None)

    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument(
        "--skip-advanced-evaluation",
        action="store_true",
    )
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-uncertainty", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("AI ECG Master Agent Started")
    print(f"Project root: {PROJECT_ROOT}")

    planner = PlannerAgent(
        PlannerConfig(
            project_root=str(PROJECT_ROOT),
            device=args.device,
            checkpoint=args.checkpoint,
            mc_samples=args.mc_samples,
            continue_on_error=args.continue_on_error,
            run_scribe=args.run_scribe,
            scribe_path=args.scribe_path,
            record_name=args.record_name,
            clinical_text=args.clinical_text,
            run_shap=not args.skip_shap,
            run_advanced_evaluation=(
                not args.skip_advanced_evaluation
            ),
            run_calibration=not args.skip_calibration,
            run_uncertainty=not args.skip_uncertainty,
        )
    )

    result = planner.run()

    print("\nAI ECG Master Agent Finished")
    print(f"Checkpoint: {result['checkpoint']}")
    print(
        "Manifest: "
        f"{result['report'].get('manifest', 'not generated')}"
    )


if __name__ == "__main__":
    main()
