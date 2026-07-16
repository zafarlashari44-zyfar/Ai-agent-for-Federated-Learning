from __future__ import annotations

import os
from typing import Any

from flwr.app import Context


ENV_PREFIX = "FL_ECG_"


def _environment_key(name: str) -> str:
    return (
        ENV_PREFIX
        + name.upper().replace("-", "_")
    )


def get_runtime_value(
    context: Context,
    name: str,
    default: Any,
) -> Any:
    environment_value = os.getenv(
        _environment_key(name)
    )

    if environment_value is None:
        return context.run_config.get(
            name,
            default,
        )

    if isinstance(default, bool):
        return environment_value.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    if isinstance(default, int):
        return int(environment_value)

    if isinstance(default, float):
        return float(environment_value)

    return environment_value