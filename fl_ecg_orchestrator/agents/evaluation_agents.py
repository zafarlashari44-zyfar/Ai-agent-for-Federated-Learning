from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .core import BaseAgent, CommandAgent


class ScriptEvaluationAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        script_name: str,
        device: str,
        extra_arguments: list[str] | None = None,
    ) -> None:
        super().__init__(name)
        self.script_name = script_name
        self.device = device
        self.extra_arguments = extra_arguments or []

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        project_root = Path(context["project_root"])
        checkpoint = context["checkpoint"]
        script = project_root / "scripts" / self.script_name

        if not script.exists():
            raise FileNotFoundError(f"Missing script: {script}")

        arguments = [
            "--device",
            self.device,
            "--checkpoint",
            checkpoint,
            *self.extra_arguments,
        ]

        return CommandAgent(
            name=self.name,
            script_path=Path("scripts") / self.script_name,
            arguments=arguments,
        ).execute(context)


class ShapAgent(ScriptEvaluationAgent):
    def __init__(self, device: str) -> None:
        super().__init__(
            "shap_agent",
            "run_shap.py",
            device,
        )


class AdvancedEvaluationAgent(ScriptEvaluationAgent):
    def __init__(self, device: str) -> None:
        super().__init__(
            "advanced_evaluation_agent",
            "run_advanced_evaluation.py",
            device,
        )


class CalibrationAgent(ScriptEvaluationAgent):
    def __init__(self, device: str) -> None:
        super().__init__(
            "calibration_agent",
            "run_calibration.py",
            device,
        )


class UncertaintyAgent(ScriptEvaluationAgent):
    def __init__(self, device: str, mc_samples: int) -> None:
        super().__init__(
            "uncertainty_agent",
            "run_uncertainty.py",
            device,
            ["--mc-samples", str(mc_samples)],
        )
