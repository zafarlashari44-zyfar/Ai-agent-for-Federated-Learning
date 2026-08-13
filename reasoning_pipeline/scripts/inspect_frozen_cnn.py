from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


def load_checkpoint(path: Path) -> Any:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in (
            "model_state_dict",
            "state_dict",
            "global_model_state_dict",
            "net",
            "model",
        ):
            candidate = checkpoint.get(key)

            if isinstance(candidate, Mapping):
                return candidate

        if checkpoint and all(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        ):
            return checkpoint

    raise ValueError(
        "The checkpoint does not contain a recognisable PyTorch state dictionary."
    )


def describe_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    print("\nCheckpoint tensors")
    print("=" * 80)

    for name, tensor in state_dict.items():
        print(
            f"{name:<60} "
            f"shape={tuple(tensor.shape)} "
            f"dtype={tensor.dtype}"
        )


def describe_model(model: nn.Module) -> None:
    print("\nModel")
    print("=" * 80)
    print(model)

    print("\nNamed modules")
    print("=" * 80)

    final_conv_name: str | None = None
    final_conv: nn.Conv1d | None = None

    for name, module in model.named_modules():
        if not name:
            continue

        print(f"{name:<60} {module.__class__.__name__}")

        if isinstance(module, nn.Conv1d):
            final_conv_name = name
            final_conv = module

    if final_conv_name is None or final_conv is None:
        print("\nNo Conv1d layer was found.")
        return

    print("\nCandidate Grad CAM target")
    print("=" * 80)
    print(f"name={final_conv_name}")
    print(f"module={final_conv}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    checkpoint = load_checkpoint(args.checkpoint)

    if isinstance(checkpoint, nn.Module):
        describe_model(checkpoint)
        return

    state_dict = extract_state_dict(checkpoint)
    describe_state_dict(state_dict)

    print(
        "\nThe checkpoint contains weights only. "
        "The Python class that defines the CNN is still required "
        "before Grad CAM can be connected safely."
    )


if __name__ == "__main__":
    main()
