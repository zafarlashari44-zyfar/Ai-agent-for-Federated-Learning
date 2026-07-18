from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .core import BaseAgent, save_json


class ReportAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__("report_agent")

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        project_root = Path(context["project_root"])
        output_dir = (
            project_root
            / "outputs"
            / "agentic_run"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        discovered = {}
        candidate_files = {
            "advanced_evaluation": (
                project_root
                / "outputs"
                / "evaluation"
                / "advanced_evaluation_metrics.json"
            ),
            "calibration": (
                project_root
                / "outputs"
                / "calibration"
                / "calibration_metrics.json"
            ),
            "uncertainty": (
                project_root
                / "outputs"
                / "uncertainty"
                / "uncertainty_metrics.json"
            ),
        }

        for name, path in candidate_files.items():
            if path.exists():
                try:
                    discovered[name] = json.loads(
                        path.read_text(encoding="utf-8")
                    )
                except json.JSONDecodeError:
                    discovered[name] = {
                        "warning": f"Invalid JSON in {path}",
                    }

        manifest = {
            "project": "Agentic Federated ECG Analysis",
            "checkpoint": context.get("checkpoint"),
            "agent_results": [
                result.to_dict()
                for result in context.get("results", [])
            ],
            "research_results": discovered,
        }

        manifest_path = output_dir / "agent_run_manifest.json"
        save_json(manifest_path, manifest)

        lines = [
            "Agentic Federated ECG Analysis",
            "=" * 31,
            "",
            f"Checkpoint: {context.get('checkpoint', 'not resolved')}",
            "",
            "Agent execution:",
        ]

        for result in context.get("results", []):
            lines.append(
                f"- {result.name}: {result.status} "
                f"({result.duration_seconds:.2f}s)"
            )

        if discovered:
            lines.extend(["", "Research outputs found:"])
            for name in discovered:
                lines.append(f"- {name}")

        summary_path = output_dir / "agent_run_summary.txt"
        summary_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        return {
            "manifest": str(manifest_path),
            "summary": str(summary_path),
            "research_outputs": list(discovered),
        }
