import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperimentLogger:

    def __init__(
        self,
        experiment_name: str,
        output_root: str = "fl_ecg_orchestrator/outputs",
    ):
        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")

        self.run_id = (
            f"{experiment_name}_{timestamp}"
        )

        self.run_dir = (
            Path(output_root)
            / "metrics"
            / self.run_id
        )

        self.run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def write_json(
        self,
        filename: str,
        payload: dict[str, Any],
    ) -> Path:
        output_path = self.run_dir / filename

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
            )

        return output_path

    def write_metadata(
        self,
        payload: dict[str, Any],
    ) -> Path:
        return self.write_json(
            "metadata.json",
            payload,
        )

    def write_summary(
        self,
        payload: dict[str, Any],
    ) -> Path:
        return self.write_json(
            "summary.json",
            payload,
        )
