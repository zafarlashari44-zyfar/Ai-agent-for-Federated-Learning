from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from .core import BaseAgent


class ScribeAgent(BaseAgent):
    """Adapter around the teammate's existing agent1_scribe.py.

    The teammate's implementation remains the source of truth. This adapter
    imports it without rewriting or taking ownership of that contribution.
    """

    def __init__(
        self,
        scribe_path: str = "agent1_scribe.py",
        record_name: str | None = None,
        clinical_text: str | None = None,
    ) -> None:
        super().__init__("scribe_agent")
        self.scribe_path = scribe_path
        self.record_name = record_name
        self.clinical_text = clinical_text

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        project_root = Path(context["project_root"])
        module_path = project_root / self.scribe_path

        if not module_path.exists():
            return {
                "skipped": True,
                "reason": (
                    f"Teammate Scribe file not found at {module_path}. "
                    "Place agent1_scribe.py in the project root or pass "
                    "--scribe-path."
                ),
            }

        if self.record_name is None and self.clinical_text is None:
            return {
                "skipped": True,
                "reason": (
                    "Scribe agent is installed and preserved, but no raw "
                    "record or clinical text was supplied for this run."
                ),
                "module": str(module_path),
            }

        spec = importlib.util.spec_from_file_location(
            "teammate_agent1_scribe",
            module_path,
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        signal_result = None
        entity_result = None
        fused_result = None

        if self.record_name is not None:
            if not hasattr(module, "process_ecg_pipeline"):
                raise AttributeError(
                    "agent1_scribe.py has no process_ecg_pipeline function."
                )
            signal_result = module.process_ecg_pipeline(self.record_name)

        if self.clinical_text is not None:
            if not hasattr(module, "extract_clinical_entities"):
                raise AttributeError(
                    "agent1_scribe.py has no extract_clinical_entities function."
                )
            entity_result = module.extract_clinical_entities(
                self.clinical_text
            )

        if (
            signal_result is not None
            and entity_result is not None
            and hasattr(module, "fuse_agent1_outputs")
        ):
            fused_result = module.fuse_agent1_outputs(
                signal_result,
                entity_result,
            )

        return {
            "module": str(module_path),
            "signal_result": self._serializable(signal_result),
            "entity_result": self._serializable(entity_result),
            "fused_result": self._serializable(fused_result),
        }

    def _serializable(self, value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if hasattr(value, "to_dict"):
            return value.to_dict()
        return value
