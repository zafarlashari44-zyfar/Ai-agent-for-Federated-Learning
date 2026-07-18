from __future__ import annotations

from pathlib import Path
from typing import Any

from .core import BaseAgent


class CheckpointAgent(BaseAgent):
    def __init__(self, explicit_checkpoint: str | None = None) -> None:
        super().__init__("checkpoint_agent")
        self.explicit_checkpoint = explicit_checkpoint

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        project_root = Path(context["project_root"])

        if self.explicit_checkpoint:
            checkpoint = Path(self.explicit_checkpoint).expanduser().resolve()
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint}"
                )
        else:
            locations = [
                project_root
                / "fl_ecg_orchestrator"
                / "outputs"
                / "checkpoints",
                project_root / "outputs" / "checkpoints",
            ]
            checkpoints: list[Path] = []
            for location in locations:
                if location.exists():
                    checkpoints.extend(location.glob("*.pth"))

            if not checkpoints:
                raise FileNotFoundError(
                    "No .pth checkpoint found. Train the model first or "
                    "pass --checkpoint."
                )

            final_models = [
                item
                for item in checkpoints
                if "final" in item.name.lower()
            ]
            checkpoint = max(
                final_models or checkpoints,
                key=lambda item: item.stat().st_mtime,
            )

        context["checkpoint"] = str(checkpoint)
        return {
            "checkpoint": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
        }
