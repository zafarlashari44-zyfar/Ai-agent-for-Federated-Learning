from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from research.detailed_arrhythmia.dataset.annotations import patient_id_for_record


@dataclass(frozen=True)
class SplitManifest:
    seed: int
    train_records: tuple[str, ...]
    validation_records: tuple[str, ...]
    test_records: tuple[str, ...]

    def __post_init__(self) -> None:
        patient_sets = [
            {patient_id_for_record(item) for item in records}
            for records in (
                self.train_records,
                self.validation_records,
                self.test_records,
            )
        ]
        leakage = any(
            patient_sets[i] & patient_sets[j]
            for i in range(3)
            for j in range(i + 1, 3)
        )
        if leakage:
            raise ValueError("Patient leakage detected across dataset splits")

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "train_records": self.train_records,
            "validation_records": self.validation_records,
            "test_records": self.test_records,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def create_patient_independent_split(
    record_ids: tuple[str, ...],
    *,
    seed: int = 42,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.20,
    record_label_counts: dict[str, dict[str, int]] | None = None,
    required_labels: tuple[str, ...] = (),
    search_iterations: int = 5000,
) -> SplitManifest:
    grouped: dict[str, list[str]] = {}
    for record_id in record_ids:
        grouped.setdefault(patient_id_for_record(record_id), []).append(record_id)
    patients = sorted(grouped)
    test_count = max(1, round(len(patients) * test_fraction))
    validation_count = max(1, round(len(patients) * validation_fraction))
    rng = random.Random(seed)

    def score(candidate: list[str]) -> float | None:
        if record_label_counts is None or not required_labels:
            return 0.0
        groups = (
            candidate[test_count : test_count + validation_count],
            candidate[:test_count],
            candidate[test_count + validation_count :],
        )
        totals = {
            label: sum(
                record_label_counts[record][label]
                for record in record_ids
            )
            for label in required_labels
        }
        fractions = (
            validation_fraction,
            test_fraction,
            1.0 - validation_fraction - test_fraction,
        )
        value = 0.0
        for group, target_fraction in zip(groups, fractions, strict=True):
            records = [record for patient in group for record in grouped[patient]]
            for label in required_labels:
                count = sum(record_label_counts[record][label] for record in records)
                if count == 0:
                    return None
                value += abs(count / totals[label] - target_fraction)
        return value

    best: list[str] | None = None
    best_score = float("inf")
    for _ in range(search_iterations):
        candidate = patients.copy()
        rng.shuffle(candidate)
        candidate_score = score(candidate)
        if candidate_score is not None and candidate_score < best_score:
            best = candidate
            best_score = candidate_score
    if best is None:
        raise ValueError(
            "Unable to create patient-independent splits with every required "
            "class represented"
        )
    test_patients = set(best[:test_count])
    validation_patients = set(best[test_count : test_count + validation_count])
    train_patients = set(best) - test_patients - validation_patients

    def records_for(selected: set[str]) -> tuple[str, ...]:
        return tuple(sorted(item for patient in selected for item in grouped[patient]))

    return SplitManifest(
        seed=seed,
        train_records=records_for(train_patients),
        validation_records=records_for(validation_patients),
        test_records=records_for(test_patients),
    )
