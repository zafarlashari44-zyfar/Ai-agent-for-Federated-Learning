from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from research.detailed_arrhythmia.config import TrainingConfig
from research.detailed_arrhythmia.dataset.annotations import (
    extract_annotation_frequencies,
    extract_record_label_counts,
    write_frequency_reports,
)
from research.detailed_arrhythmia.dataset.beats import extract_expert_annotated_beats
from research.detailed_arrhythmia.dataset.ontology import DEFAULT_ONTOLOGY
from research.detailed_arrhythmia.dataset.splits import create_patient_independent_split
from research.detailed_arrhythmia.evaluation.ablation import compare_experiments
from research.detailed_arrhythmia.training.augmentation import SafeECGAugmenter
from research.detailed_arrhythmia.training.experiment import run_experiment
from research.detailed_arrhythmia.training.weights import calculate_class_weights

MIT_BIH_RECORDS = (
    "100", "101", "102", "103", "104", "105", "106", "107",
    "108", "109", "111", "112", "113", "114", "115", "116",
    "117", "118", "119", "121", "122", "123", "124", "200",
    "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
)


def write_augmentation_figure(
    originals: NDArray[np.float32],
    augmented: NDArray[np.float32],
    path: Path,
) -> None:
    """Save representative pairs for required human visual review."""
    import matplotlib.pyplot as plt

    pair_count = min(6, originals.shape[0])
    figure, axes = plt.subplots(pair_count, 1, figsize=(10, 2.2 * pair_count))
    axes_array = np.atleast_1d(axes)
    for index, axis in enumerate(axes_array):
        axis.plot(originals[index], label="original", linewidth=1.2)
        axis.plot(augmented[index], label="augmented", linewidth=1.0, alpha=0.8)
        axis.set_title(f"Training beat example {index + 1}")
        axis.set_xlim(0, originals.shape[1] - 1)
        axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records_dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/detailed_arrhythmia/outputs"),
    )
    arguments = parser.parse_args()
    output = arguments.output_dir
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    frequencies = extract_annotation_frequencies(arguments.records_dir, MIT_BIH_RECORDS)
    write_frequency_reports(frequencies, reports)
    record_counts = extract_record_label_counts(
        arguments.records_dir, MIT_BIH_RECORDS, DEFAULT_ONTOLOGY
    )
    manifest = create_patient_independent_split(
        MIT_BIH_RECORDS,
        seed=42,
        record_label_counts=record_counts,
        required_labels=DEFAULT_ONTOLOGY.labels,
    )
    manifest.write(output / "split_manifest.json")
    arrays: dict[str, tuple[NDArray[np.float32], NDArray[np.int64]]] = {}
    split_summaries: dict[str, Any] = {}
    summary: dict[str, Any] = {
        "split_manifest_hash": manifest.sha256,
        "splits": split_summaries,
    }
    for split, records in (
        ("train", manifest.train_records),
        ("validation", manifest.validation_records),
        ("test", manifest.test_records),
    ):
        beats, labels, sources = extract_expert_annotated_beats(
            arguments.records_dir, records, DEFAULT_ONTOLOGY
        )
        arrays[split] = (beats, labels)
        counts = np.bincount(labels, minlength=len(DEFAULT_ONTOLOGY.labels))
        split_summaries[split] = {
            "records": records,
            "patients": len(records) - int("201" in records and "202" in records),
            "beats": int(labels.size),
            "class_counts": dict(
                zip(DEFAULT_ONTOLOGY.labels, counts.tolist(), strict=True)
            ),
        }
        del sources
    config = TrainingConfig()
    weights = calculate_class_weights(
        arrays["train"][1],
        len(DEFAULT_ONTOLOGY.labels),
        method="sqrt_inverse_frequency",
    )
    summary["training_class_weights"] = dict(
        zip(DEFAULT_ONTOLOGY.labels, weights.tolist(), strict=True)
    )
    (reports / "split_and_weight_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    augmenter = SafeECGAugmenter(config.augmentation, seed=config.seed)
    example_count = min(12, arrays["train"][0].shape[0])
    originals = arrays["train"][0][:example_count]
    augmented = np.stack([augmenter(item) for item in originals])
    np.savez_compressed(
        reports / "augmentation_examples.npz",
        originals=originals,
        augmented=augmented,
    )
    write_augmentation_figure(
        originals,
        augmented,
        reports / "augmentation_examples.png",
    )
    experiment_a_path = output / "experiment_a_weighted_only" / "results.json"
    if experiment_a_path.exists():
        experiment_a = json.loads(experiment_a_path.read_text(encoding="utf-8"))
    else:
        experiment_a = run_experiment(
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
            output_dir=output / "experiment_a_weighted_only",
        )
    experiment_b_path = (
        output / "experiment_b_safe_augmentation" / "results.json"
    )
    if experiment_b_path.exists():
        experiment_b = json.loads(experiment_b_path.read_text(encoding="utf-8"))
    else:
        experiment_b = run_experiment(
            train_beats=arrays["train"][0],
            train_labels=arrays["train"][1],
            validation_beats=arrays["validation"][0],
            validation_labels=arrays["validation"][1],
            test_beats=arrays["test"][0],
            test_labels=arrays["test"][1],
            ontology=DEFAULT_ONTOLOGY,
            split_manifest=manifest,
            config=config,
            augmentation_enabled=True,
            output_dir=output / "experiment_b_safe_augmentation",
        )
    comparison = {
        "validation": compare_experiments(
            experiment_a["validation"], experiment_b["validation"]
        ),
        "test": compare_experiments(experiment_a["test"], experiment_b["test"]),
    }
    (reports / "ablation.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
