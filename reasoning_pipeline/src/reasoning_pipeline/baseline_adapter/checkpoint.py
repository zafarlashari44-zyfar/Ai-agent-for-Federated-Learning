from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from reasoning_pipeline.baseline_adapter.exceptions import (
    CheckpointNotFoundError,
    InvalidCheckpointError,
)


def calculate_sha256(checkpoint_path: str | Path) -> str:
    path = Path(checkpoint_path).expanduser().resolve()

    if not path.is_file():
        raise CheckpointNotFoundError(
            f"Checkpoint not found: {path}"
        )

    digest = hashlib.sha256()

    with path.open("rb") as checkpoint_file:
        for chunk in iter(
            lambda: checkpoint_file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    path = Path(checkpoint_path).expanduser().resolve()

    if not path.is_file():
        raise CheckpointNotFoundError(
            f"Checkpoint not found: {path}"
        )

    try:
        checkpoint = torch.load(
            path,
            map_location=device,
        )
    except Exception as exc:
        raise InvalidCheckpointError(
            f"Could not load checkpoint: {path}"
        ) from exc

    if not isinstance(checkpoint, dict):
        raise InvalidCheckpointError(
            "Checkpoint must contain a dictionary."
        )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    if not isinstance(state_dict, dict):
        raise InvalidCheckpointError(
            "Checkpoint does not contain a valid state dictionary."
        )

    try:
        model.load_state_dict(
            state_dict,
            strict=True,
        )
    except Exception as exc:
        raise InvalidCheckpointError(
            "Checkpoint is incompatible with ECGCNN1D."
        ) from exc

    model.to(device)
    model.eval()

    return checkpoint
