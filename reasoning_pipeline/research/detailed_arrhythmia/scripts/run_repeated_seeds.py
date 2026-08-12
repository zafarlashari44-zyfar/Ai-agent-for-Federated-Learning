from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from research.detailed_arrhythmia.config import TrainingConfig
from research.detailed_arrhythmia.dataset.annotations import (
    extract_record_label_counts,
)
from research.detailed_arrhythmia.dataset.beats import (
    extract_expert_annotated_beats,
)
from research.detailed_arrhythmia.dataset.ontology import DEFAULT_ONTOLOGY
from research.detailed_arrhythmia.dataset.splits import (
    create_patient_independent_split,
)
from research.detailed_arrhythmia.training.experiment import run_experiment

MIT_BIH_RECORDS = (
    "100", "101", "102", "103", "104", "105", "106", "107",
    "108", "109", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "121", "122", "123", "124", "200",
    "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
)

TRAINING_SEEDS = (42, 123, 456, 789, 2026)
SPLIT_SEED = 42


def summarise_values(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)

    return {
        "values": values,
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def build_summary(results: dict[int, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "experiment": "repeated_seeds_weighted_cross_entropy",
        "split_seed": SPLIT_SEED,
        "training_seeds": list(TRAINING_SEEDS),
        "number_of_runs": len(results),
        "validation": {},
        "test": {},
    }

    for split in ("validation", "test"):
        split_summary: dict[str, Any] = {}

        for metric in (
            "macro_f1",
            "weighted_f1",
            "balanced_accuracy",
            "expected_calibration_error_10_bins",
        ):
            values = [
                float(results[seed][split][metric])
                for seed in TRAINING_SEEDS
            ]

            split_summary[metric] = summarise_values(values)

        per_class: dict[str, Any] = {}

        for label in DEFAULT_ONTOLOGY.labels:
            class_summary: dict[str, Any] = {}

            for metric in ("precision", "recall", "f1"):
                values = [
                    float(
                        results[seed][split]["per_class"][label][metric]
                    )
                    for seed in TRAINING_SEEDS
                ]

                class_summary[metric] = summarise_values(values)

            per_class[label] = class_summary

        split_summary["per_class"] = per_class
        summary[split] = split_summary

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--records-dir",
        type=Path,
        required=True,
        help="Path to the raw MIT-BIH WFDB records directory.",
    )

    arguments = parser.parse_args()

    records_dir = arguments.records_dir.expanduser().resolve()

    if not records_dir.exists():
        raise FileNotFoundError(
            f"MIT-BIH dataset directory not found: {records_dir}"
        )

    output_dir = (
        Path("research")
        / "detailed_arrhythmia"
        / "outputs"
        / "repeated_seeds_weighted_ce"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("Experiment C")
    print("Repeated training seeds with fixed patient split")
    print("=" * 70)
    print(f"Records directory: {records_dir}")
    print(f"Split seed: {SPLIT_SEED}")
    print(f"Training seeds: {TRAINING_SEEDS}")
    print()

    print("Preparing fixed patient-independent split...")

    record_counts = extract_record_label_counts(
        records_dir,
        MIT_BIH_RECORDS,
        DEFAULT_ONTOLOGY,
    )

    manifest = create_patient_independent_split(
        MIT_BIH_RECORDS,
        seed=SPLIT_SEED,
        record_label_counts=record_counts,
        required_labels=DEFAULT_ONTOLOGY.labels,
    )

    manifest.write(output_dir / "split_manifest.json")

    print(f"Split hash: {manifest.sha256}")
    print()

    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for split_name, records in (
        ("train", manifest.train_records),
        ("validation", manifest.validation_records),
        ("test", manifest.test_records),
    ):
        print(f"Loading {split_name} beats...")

        beats, labels, _ = extract_expert_annotated_beats(
            records_dir,
            records,
            DEFAULT_ONTOLOGY,
        )

        arrays[split_name] = (beats, labels)

        class_counts = np.bincount(
            labels,
            minlength=len(DEFAULT_ONTOLOGY.labels),
        )

        print(
            f"{split_name}: "
            f"{len(records)} records, "
            f"{labels.size} beats"
        )

        print(
            "Class counts: "
            + ", ".join(
                f"{label}={int(count)}"
                for label, count in zip(
                    DEFAULT_ONTOLOGY.labels,
                    class_counts,
                    strict=True,
                )
            )
        )

        print()

    base_config = TrainingConfig()

    results: dict[int, dict[str, Any]] = {}

    for seed in TRAINING_SEEDS:
        seed_output = output_dir / f"seed_{seed}"
        result_path = seed_output / "results.json"

        print()
        print("=" * 70)
        print(f"Training seed {seed}")
        print("=" * 70)

        if result_path.exists():
            print(f"Existing result found for seed {seed}. Reusing it.")

            result = json.loads(
                result_path.read_text(encoding="utf-8")
            )

        else:
            config = replace(
                base_config,
                seed=seed,
            )

            result = run_experiment(
                train_beats=arrays["train"][0],
                train_labels=arrays["train"][1],
                validation_beats=arrays["validation"][0],
                validation_labels=arrays["validation"][1],
                test_beats=arrays["test"][0],
                test_labels=arrays["test"][1],
                ontology=DEFAULT_ONTOLOGY,
                split_manifest=manifest,
                config=config,
                augmentation_enabled=False,
                output_dir=seed_output,
            )

        results[seed] = result

        print(
            f"Seed {seed} | "
            f"Validation Macro F1: "
            f"{result['validation']['macro_f1']:.4f} | "
            f"Test Macro F1: "
            f"{result['test']['macro_f1']:.4f}"
        )

    summary = build_summary(results)

    summary_path = output_dir / "summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("Repeated seed experiment complete")
    print("=" * 70)

    print(
        "Validation Macro F1: "
        f"{summary['validation']['macro_f1']['mean']:.4f} "
        "± "
        f"{summary['validation']['macro_f1']['std']:.4f}"
    )

    print(
        "Test Macro F1: "
        f"{summary['test']['macro_f1']['mean']:.4f} "
        "± "
        f"{summary['test']['macro_f1']['std']:.4f}"
    )

    print()
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()