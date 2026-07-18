from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentResult:
    name: str
    status: str
    started_at: float
    finished_at: float
    output: dict[str, Any] = field(default_factory=dict)
    error: dict[str, str] | None = None

    @property
    def duration_seconds(self) -> float:
        return self.finished_at - self.started_at

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["duration_seconds"] = self.duration_seconds
        return payload


class BaseAgent(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    def run(self, context: dict[str, Any]) -> AgentResult:
        started = time.time()
        try:
            output = self.execute(context)
            return AgentResult(
                name=self.name,
                status="success",
                started_at=started,
                finished_at=time.time(),
                output=output or {},
            )
        except Exception as exc:
            return AgentResult(
                name=self.name,
                status="failed",
                started_at=started,
                finished_at=time.time(),
                error={
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )

    @abstractmethod
    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class CommandAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        script_path: Path,
        arguments: list[str] | None = None,
        required: bool = True,
    ) -> None:
        super().__init__(name)
        self.script_path = script_path
        self.arguments = arguments or []
        self.required = required

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        project_root = Path(context["project_root"])
        script = project_root / self.script_path

        if not script.exists():
            message = f"Script not found: {script}"
            if self.required:
                raise FileNotFoundError(message)
            return {"skipped": True, "reason": message}

        command = [sys.executable, str(script), *self.arguments]
        process = subprocess.run(
            command,
            cwd=project_root,
            text=True,
            capture_output=True,
            check=False,
        )

        output = {
            "command": command,
            "return_code": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }

        if process.returncode != 0:
            raise RuntimeError(
                f"{self.name} failed with exit code {process.returncode}.\n"
                f"STDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"
            )

        return output


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, default=str)
