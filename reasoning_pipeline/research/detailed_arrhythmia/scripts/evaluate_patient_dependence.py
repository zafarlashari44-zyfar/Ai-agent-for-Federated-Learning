import argparse
import json
from pathlib import Path

from research.detailed_arrhythmia.dataset.beats import extract_expert_annotated_beats
from research.detailed_arrhythmia.dataset.ontology import DEFAULT_ONTOLOGY
from research.detailed_arrhythmia.dataset.splits import SplitManifest
from research.detailed_arrhythmia.evaluation.patient_dependence import (
    evaluate_checkpoint_by_record,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records_dir", type=Path)
    parser.add_argument("outputs_dir", type=Path)
    arguments = parser.parse_args()
    payload = json.loads(
        (arguments.outputs_dir / "split_manifest.json").read_text(encoding="utf-8")
    )
    manifest = SplitManifest(
        seed=payload["seed"],
        train_records=tuple(payload["train_records"]),
        validation_records=tuple(payload["validation_records"]),
        test_records=tuple(payload["test_records"]),
    )
    output: dict[str, object] = {}
    for split, records in (
        ("validation", manifest.validation_records),
        ("test", manifest.test_records),
    ):
        beats, labels, sources = extract_expert_annotated_beats(
            arguments.records_dir, records, DEFAULT_ONTOLOGY
        )
        for experiment in (
            "experiment_a_weighted_only",
            "experiment_b_safe_augmentation",
        ):
            output[f"{experiment}_{split}"] = evaluate_checkpoint_by_record(
                arguments.outputs_dir / experiment / "detailed_classifier.pt",
                beats,
                labels,
                sources,
                DEFAULT_ONTOLOGY.labels,
            )
    destination = arguments.outputs_dir / "reports" / "per_record_metrics.json"
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
