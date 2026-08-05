from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckpointMetadata:
    model_version: str
    ontology_version: str
    labels: tuple[str, ...]
    split_manifest_hash: str
    training_seed: int
    class_weight_method: str
    class_weights: tuple[float, ...]
    augmentation_configuration: dict[str, Any] | None
    checkpoint_hash: str

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
