from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .reasoning_agent import ReasoningAgent
from .checkpoint_agent import CheckpointAgent
from .core import AgentResult
from .evaluation_agents import (
    AdvancedEvaluationAgent,
    CalibrationAgent,
    ShapAgent,
    UncertaintyAgent,
)
from .report_agent import ReportAgent
from .scribe_agent import ScribeAgent


@dataclass
class PlannerConfig:
    project_root: str
    device: str = "cpu"
    checkpoint: str | None = None
    mc_samples: int = 30
    continue_on_error: bool = False
    run_scribe: bool = False
    scribe_path: str = "agent1_scribe.py"
    record_name: str | None = None
    clinical_text: str | None = None
    run_shap: bool = True
    run_advanced_evaluation: bool = True
    run_calibration: bool = True
    run_uncertainty: bool = True
    run_reasoning: bool = True
class PlannerAgent:
    """Master agent that delegates work to specialized sub-agents."""

    def __init__(self, config: PlannerConfig) -> None:
        self.config = config

    def run(self) -> dict[str, Any]:
        context: dict[str, Any] = {
            "project_root": str(
                Path(self.config.project_root).resolve()
            ),
            "results": [],
        }

        agents = []

        if self.config.run_scribe:
            agents.append(
                ScribeAgent(
                    scribe_path=self.config.scribe_path,
                    record_name=self.config.record_name,
                    clinical_text=self.config.clinical_text,
                )
            )

        agents.append(
            CheckpointAgent(
                explicit_checkpoint=self.config.checkpoint
            )
        )

        if self.config.run_shap:
            agents.append(ShapAgent(self.config.device))

        if self.config.run_advanced_evaluation:
            agents.append(
                AdvancedEvaluationAgent(self.config.device)
            )

        if self.config.run_calibration:
            agents.append(CalibrationAgent(self.config.device))

        if self.config.run_uncertainty:
            agents.append(
                UncertaintyAgent(
                    self.config.device,
                    self.config.mc_samples,
                )
            )

        if self.config.run_reasoning:
            agents.append(
                ReasoningAgent()
            )

        for agent in agents:
            print(f"\n[{agent.name}] started")
            result = agent.run(context)
            context["results"].append(result)
            print(
                f"[{agent.name}] {result.status} "
                f"in {result.duration_seconds:.2f}s"
            )

            if (
                result.status == "failed"
                and not self.config.continue_on_error
            ):
                print(
                    "Planner stopped because an agent failed. "
                    "Use --continue-on-error to run remaining agents."
                )
                break

        report_result = ReportAgent().run(context)
        context["results"].append(report_result)

        return {
            "checkpoint": context.get("checkpoint"),
            "results": [
                item.to_dict()
                for item in context["results"]
            ],
            "report": report_result.output,
        }


