from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.detailed_arrhythmia.dataset.beats import (
    extract_expert_annotated_beats,
)
from research.detailed_arrhythmia.dataset.ontology import DEFAULT_ONTOLOGY
from research.detailed_arrhythmia.dataset.splits import SplitManifest
from research.detailed_arrhythmia.evaluation.patient_dependence import (
    evaluate_checkpoint_by_patient,
    evaluate_checkpoint_by_record,
)

SEEDS = (42, 123, 456, 789, 2026)


def load_manifest(path: Path) -> SplitManifest:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    return SplitManifest(
        seed=int(payload["seed"]),
        train_records=tuple(payload["train_records"]),
        validation_records=tuple(payload["validation_records"]),
        test_records=tuple(payload["test_records"]),
    )


def summarise(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)

    return {
        "values": values,
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def aggregate_groups(
    runs: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    groups = sorted(
        {
            group
            for seed_results in runs.values()
            for group in seed_results
        }
    )

    output: dict[str, Any] = {}

    for group in groups:
        available = [
            runs[seed][group]
            for seed in SEEDS
            if group in runs[seed]
        ]

        output[group] = {
            "number_of_runs": len(available),
            "total_beats": available[0]["total_beats"],
            "abnormal_support": available[0]["abnormal_support"],
            "macro_f1": summarise(
                [
                    float(result["macro_f1"])
                    for result in available
                ]
            ),
            "macro_f1_present_classes": summarise(
                [
                    float(result["macro_f1_present_classes"])
                    for result in available
                ]
            ),
            "balanced_accuracy": summarise(
                [
                    float(result["balanced_accuracy"])
                    for result in available
                ]
            ),
            "abnormal_to_normal_fraction": summarise(
                [
                    float(
                        result["abnormal_to_normal_fraction"]
                    )
                    for result in available
                ]
            ),
            "per_class": {},
        }

        for label in DEFAULT_ONTOLOGY.labels:
            output[group]["per_class"][label] = {
                "support": available[0]["per_class"][label]["support"],
                "precision": summarise(
                    [
                        float(
                            result["per_class"][label]["precision"]
                        )
                        for result in available
                    ]
                ),
                "recall": summarise(
                    [
                        float(
                            result["per_class"][label]["recall"]
                        )
                        for result in available
                    ]
                ),
                "f1": summarise(
                    [
                        float(
                            result["per_class"][label]["f1"]
                        )
                        for result in available
                    ]
                ),
            }

    return output


def rank_groups(
    groups: dict[str, Any],
) -> list[dict[str, Any]]:
    ranking: list[dict[str, Any]] = []

    for group, metrics in groups.items():
        ranking.append(
            {
                "id": group,
                "total_beats": metrics["total_beats"],
                "abnormal_support": metrics["abnormal_support"],
                "macro_f1_mean": metrics["macro_f1"]["mean"],
                "macro_f1_std": metrics["macro_f1"]["std"],
                "macro_f1_present_classes_mean": (
                    metrics["macro_f1_present_classes"]["mean"]
                ),
                "macro_f1_present_classes_std": (
                    metrics["macro_f1_present_classes"]["std"]
                ),
                "abnormal_to_normal_mean": (
                    metrics["abnormal_to_normal_fraction"]["mean"]
                ),
            }
        )

    return sorted(
        ranking,
        key=lambda item: item["macro_f1_present_classes_mean"],
    )


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

    arguments = parser.parse_args()

    records_dir = arguments.records_dir.expanduser().resolve()
    input_dir = arguments.input_dir.resolve()

    manifest = load_manifest(
        input_dir / "split_manifest.json"
    )

    print()
    print("=" * 70)
    print("Per record and per patient error analysis")
    print("=" * 70)

    print("Loading fixed test split...")

    beats, targets, sources = extract_expert_annotated_beats(
        records_dir,
        manifest.test_records,
        DEFAULT_ONTOLOGY,
    )

    print(f"Test records: {len(manifest.test_records)}")
    print(f"Test beats: {targets.size}")
    print()

    record_runs: dict[int, dict[str, dict[str, Any]]] = {}
    patient_runs: dict[int, dict[str, dict[str, Any]]] = {}

    for seed in SEEDS:
        checkpoint = (
            input_dir
            / f"seed_{seed}"
            / "detailed_classifier.pt"
        )

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}"
            )

        print(f"Evaluating seed {seed}...")

        record_runs[seed] = evaluate_checkpoint_by_record(
            checkpoint,
            beats,
            targets,
            sources,
            DEFAULT_ONTOLOGY.labels,
        )

        patient_runs[seed] = evaluate_checkpoint_by_patient(
            checkpoint,
            beats,
            targets,
            sources,
            DEFAULT_ONTOLOGY.labels,
        )

    record_summary = aggregate_groups(record_runs)
    patient_summary = aggregate_groups(patient_runs)

    record_ranking_worst = rank_groups(record_summary)
    patient_ranking_worst = rank_groups(patient_summary)

    record_ranking_best = list(
        reversed(record_ranking_worst)
    )

    patient_ranking_best = list(
        reversed(patient_ranking_worst)
    )

    records_with_abnormal_beats = [
        item
        for item in record_ranking_worst
        if item["abnormal_support"] > 0
    ]

    normal_only_records = [
        item
        for item in record_ranking_worst
        if item["abnormal_support"] == 0
    ]

    patients_with_abnormal_beats = [
        item
        for item in patient_ranking_worst
        if item["abnormal_support"] > 0
    ]

    normal_only_patients = [
        item
        for item in patient_ranking_worst
        if item["abnormal_support"] == 0
    ]

    report = {
        "experiment": "per_record_and_patient_error_analysis",
        "seeds": list(SEEDS),
        "split_seed": manifest.seed,
        "test_records": list(manifest.test_records),
        "records": record_summary,
        "patients": patient_summary,
        "record_ranking_worst_to_best": record_ranking_worst,
        "record_ranking_best_to_worst": record_ranking_best,
        "patient_ranking_worst_to_best": patient_ranking_worst,
        "patient_ranking_best_to_worst": patient_ranking_best,
        "records_with_abnormal_beats": records_with_abnormal_beats,
        "normal_only_records": normal_only_records,
        "patients_with_abnormal_beats": patients_with_abnormal_beats,
        "normal_only_patients": normal_only_patients,
    }

    output_dir = input_dir / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir
        / "per_record_patient_analysis.json"
    )

    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("Worst test records with abnormal beats")
    print("=" * 70)

    for item in records_with_abnormal_beats[:10]:
        print(
            f"Record {item['id']} | "
            f"Present class Macro F1 "
            f"{item['macro_f1_present_classes_mean']:.4f} "
            f"± "
            f"{item['macro_f1_present_classes_std']:.4f} | "
            f"Abnormal to N "
            f"{item['abnormal_to_normal_mean']:.2%} | "
            f"Abnormal support "
            f"{item['abnormal_support']}"
        )

    print()
    print("=" * 70)
    print("Best test records with abnormal beats")
    print("=" * 70)

    best_abnormal_records = [
        item
        for item in record_ranking_best
        if item["abnormal_support"] > 0
    ]

    for item in best_abnormal_records[:10]:
        print(
            f"Record {item['id']} | "
            f"Present class Macro F1 "
            f"{item['macro_f1_present_classes_mean']:.4f} "
            f"± "
            f"{item['macro_f1_present_classes_std']:.4f} | "
            f"Abnormal to N "
            f"{item['abnormal_to_normal_mean']:.2%} | "
            f"Abnormal support "
            f"{item['abnormal_support']}"
        )

    if normal_only_records:
        print()
        print("=" * 70)
        print("Normal only test records")
        print("=" * 70)

        for item in normal_only_records:
            print(
                f"Record {item['id']} | "
                f"Present class Macro F1 "
                f"{item['macro_f1_present_classes_mean']:.4f} "
                f"± "
                f"{item['macro_f1_present_classes_std']:.4f}"
            )

    print()
    print("=" * 70)
    print("Worst test patients with abnormal beats")
    print("=" * 70)

    for item in patients_with_abnormal_beats[:10]:
        print(
            f"Patient {item['id']} | "
            f"Present class Macro F1 "
            f"{item['macro_f1_present_classes_mean']:.4f} "
            f"± "
            f"{item['macro_f1_present_classes_std']:.4f} | "
            f"Abnormal to N "
            f"{item['abnormal_to_normal_mean']:.2%} | "
            f"Abnormal support "
            f"{item['abnormal_support']}"
        )

    print()
    print("=" * 70)
    print("Best test patients with abnormal beats")
    print("=" * 70)

    best_abnormal_patients = [
        item
        for item in patient_ranking_best
        if item["abnormal_support"] > 0
    ]

    for item in best_abnormal_patients[:10]:
        print(
            f"Patient {item['id']} | "
            f"Present class Macro F1 "
            f"{item['macro_f1_present_classes_mean']:.4f} "
            f"± "
            f"{item['macro_f1_present_classes_std']:.4f} | "
            f"Abnormal to N "
            f"{item['abnormal_to_normal_mean']:.2%} | "
            f"Abnormal support "
            f"{item['abnormal_support']}"
        )

    if normal_only_patients:
        print()
        print("=" * 70)
        print("Normal only test patients")
        print("=" * 70)

        for item in normal_only_patients:
            print(
                f"Patient {item['id']} | "
                f"Present class Macro F1 "
                f"{item['macro_f1_present_classes_mean']:.4f} "
                f"± "
                f"{item['macro_f1_present_classes_std']:.4f}"
            )

    print()
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()