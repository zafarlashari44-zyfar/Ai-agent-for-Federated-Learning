from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from research.detailed_arrhythmia.dataset.annotations import (
    extract_record_label_counts,
    patient_id_for_record,
)
from research.detailed_arrhythmia.dataset.ontology import DEFAULT_ONTOLOGY

N_FOLDS = 5
SEARCH_SEED = 42
SEARCH_ITERATIONS = 20000


def load_split_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def group_records_by_patient(
    records: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}

    for record in records:
        patient_id = patient_id_for_record(record)

        grouped.setdefault(
            patient_id,
            [],
        ).append(record)

    return {
        patient: tuple(sorted(patient_records))
        for patient, patient_records in sorted(grouped.items())
    }


def calculate_fold_counts(
    patients: list[str],
    grouped_records: dict[str, tuple[str, ...]],
    record_counts: dict[str, dict[str, int]],
) -> dict[str, int]:
    counts = {
        label: 0
        for label in DEFAULT_ONTOLOGY.labels
    }

    for patient in patients:
        for record in grouped_records[patient]:
            for label in DEFAULT_ONTOLOGY.labels:
                counts[label] += record_counts[record][label]

    return counts


def score_folds(
    folds: list[list[str]],
    grouped_records: dict[str, tuple[str, ...]],
    record_counts: dict[str, dict[str, int]],
    development_totals: dict[str, int],
) -> float:
    expected_fraction = 1.0 / N_FOLDS

    score = 0.0

    for fold in folds:
        counts = calculate_fold_counts(
            fold,
            grouped_records,
            record_counts,
        )

        for label in DEFAULT_ONTOLOGY.labels:
            total = development_totals[label]

            if total == 0:
                continue

            observed_fraction = counts[label] / total

            score += abs(
                observed_fraction - expected_fraction
            )

    return score


def create_candidate_folds(
    patients: list[str],
    rng: random.Random,
) -> list[list[str]]:
    candidate = patients.copy()
    rng.shuffle(candidate)

    folds = [
        []
        for _ in range(N_FOLDS)
    ]

    for index, patient in enumerate(candidate):
        folds[index % N_FOLDS].append(patient)

    return folds


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--records-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=(
            Path("research")
            / "detailed_arrhythmia"
            / "outputs"
            / "repeated_seeds_weighted_ce"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path("research")
            / "detailed_arrhythmia"
            / "outputs"
            / "grouped_patient_cv"
        ),
    )

    arguments = parser.parse_args()

    records_dir = arguments.records_dir.expanduser().resolve()
    input_dir = arguments.input_dir.resolve()
    output_dir = arguments.output_dir.resolve()

    manifest = load_split_manifest(
        input_dir / "split_manifest.json"
    )

    train_records = tuple(
        manifest["train_records"]
    )

    validation_records = tuple(
        manifest["validation_records"]
    )

    locked_test_records = tuple(
        manifest["test_records"]
    )

    development_records = tuple(
        sorted(
            set(train_records)
            | set(validation_records)
        )
    )

    overlap = set(development_records) & set(
        locked_test_records
    )

    if overlap:
        raise RuntimeError(
            "Locked test leakage detected: "
            f"{sorted(overlap)}"
        )

    grouped_records = group_records_by_patient(
        development_records
    )

    development_patients = list(
        grouped_records
    )

    locked_test_patients = {
        patient_id_for_record(record)
        for record in locked_test_records
    }

    development_patient_set = set(
        development_patients
    )

    patient_overlap = (
        development_patient_set
        & locked_test_patients
    )

    if patient_overlap:
        raise RuntimeError(
            "Patient leakage into locked test cohort: "
            f"{sorted(patient_overlap)}"
        )

    record_counts = extract_record_label_counts(
        records_dir,
        development_records,
        DEFAULT_ONTOLOGY,
    )

    development_totals = {
        label: sum(
            record_counts[record][label]
            for record in development_records
        )
        for label in DEFAULT_ONTOLOGY.labels
    }

    print()
    print("=" * 70)
    print("Experiment D")
    print("Prepare grouped patient cross validation")
    print("=" * 70)

    print(
        f"Development records: "
        f"{len(development_records)}"
    )

    print(
        f"Development patients: "
        f"{len(development_patients)}"
    )

    print(
        f"Locked test records: "
        f"{len(locked_test_records)}"
    )

    print(
        f"Locked test patients: "
        f"{len(locked_test_patients)}"
    )

    print()

    print("Development class totals")

    for label in DEFAULT_ONTOLOGY.labels:
        print(
            f"{label}: "
            f"{development_totals[label]}"
        )

    rng = random.Random(
        SEARCH_SEED
    )

    best_folds: list[list[str]] | None = None
    best_score = float("inf")

    for _ in range(SEARCH_ITERATIONS):
        candidate_folds = create_candidate_folds(
            development_patients,
            rng,
        )

        candidate_score = score_folds(
            candidate_folds,
            grouped_records,
            record_counts,
            development_totals,
        )

        if candidate_score < best_score:
            best_score = candidate_score

            best_folds = [
                sorted(fold)
                for fold in candidate_folds
            ]

    if best_folds is None:
        raise RuntimeError(
            "Unable to create grouped CV folds"
        )

    fold_payloads: list[dict[str, Any]] = []

    print()
    print("=" * 70)
    print("Grouped CV folds")
    print("=" * 70)

    for fold_index, validation_patients in enumerate(
        best_folds,
        start=1,
    ):
        validation_patient_set = set(
            validation_patients
        )

        training_patients = sorted(
            development_patient_set
            - validation_patient_set
        )

        validation_records_fold = sorted(
            record
            for patient in validation_patients
            for record in grouped_records[patient]
        )

        training_records_fold = sorted(
            record
            for patient in training_patients
            for record in grouped_records[patient]
        )

        training_counts = calculate_fold_counts(
            training_patients,
            grouped_records,
            record_counts,
        )

        validation_counts = calculate_fold_counts(
            validation_patients,
            grouped_records,
            record_counts,
        )

        missing_training_labels = [
            label
            for label in DEFAULT_ONTOLOGY.labels
            if training_counts[label] == 0
        ]

        missing_validation_labels = [
            label
            for label in DEFAULT_ONTOLOGY.labels
            if validation_counts[label] == 0
        ]

        training_patient_set = set(
            training_patients
        )

        if (
            training_patient_set
            & validation_patient_set
        ):
            raise RuntimeError(
                f"Patient leakage inside fold "
                f"{fold_index}"
            )

        if (
            set(training_records_fold)
            & set(validation_records_fold)
        ):
            raise RuntimeError(
                f"Record leakage inside fold "
                f"{fold_index}"
            )

        if (
            set(training_records_fold)
            & set(locked_test_records)
        ):
            raise RuntimeError(
                f"Locked test leakage inside "
                f"fold {fold_index}"
            )

        if (
            set(validation_records_fold)
            & set(locked_test_records)
        ):
            raise RuntimeError(
                f"Locked test leakage inside "
                f"fold {fold_index}"
            )

        fold_payload = {
            "fold": fold_index,
            "training_patients": training_patients,
            "validation_patients": validation_patients,
            "training_records": training_records_fold,
            "validation_records": validation_records_fold,
            "training_class_counts": training_counts,
            "validation_class_counts": validation_counts,
            "missing_training_labels": (
                missing_training_labels
            ),
            "missing_validation_labels": (
                missing_validation_labels
            ),
        }

        fold_payloads.append(
            fold_payload
        )

        print()
        print(f"Fold {fold_index}")

        print(
            "Validation patients: "
            + ", ".join(validation_patients)
        )

        print(
            f"Training records: "
            f"{len(training_records_fold)}"
        )

        print(
            f"Validation records: "
            f"{len(validation_records_fold)}"
        )

        print(
            "Validation counts: "
            + ", ".join(
                f"{label}="
                f"{validation_counts[label]}"
                for label in DEFAULT_ONTOLOGY.labels
            )
        )

        if missing_validation_labels:
            print(
                "WARNING missing validation labels: "
                + ", ".join(
                    missing_validation_labels
                )
            )

        if missing_training_labels:
            print(
                "ERROR missing training labels: "
                + ", ".join(
                    missing_training_labels
                )
            )

    payload = {
        "experiment": "grouped_patient_cross_validation",
        "n_folds": N_FOLDS,
        "search_seed": SEARCH_SEED,
        "search_iterations": SEARCH_ITERATIONS,
        "fold_balance_score": best_score,
        "development_records": list(
            development_records
        ),
        "development_patients": sorted(
            development_patients
        ),
        "locked_test_records": list(
            locked_test_records
        ),
        "locked_test_patients": sorted(
            locked_test_patients
        ),
        "development_class_totals": (
            development_totals
        ),
        "folds": fold_payloads,
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "grouped_cv_manifest.json"
    )

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("Grouped CV preparation complete")
    print("=" * 70)

    print(
        f"Fold balance score: "
        f"{best_score:.6f}"
    )

    print(
        f"Manifest written to: "
        f"{output_path}"
    )

    print()
    print(
        "Locked test cohort was not used "
        "to construct the CV folds."
    )


if __name__ == "__main__":
    main()