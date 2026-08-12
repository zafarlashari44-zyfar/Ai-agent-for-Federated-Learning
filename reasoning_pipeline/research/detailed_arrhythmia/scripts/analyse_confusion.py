from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from research.detailed_arrhythmia.dataset.ontology import DEFAULT_ONTOLOGY
from research.detailed_arrhythmia.evaluation.confusion import (
    aggregate_confusion_matrices,
    analyse_confusion_matrix,
)

DEFAULT_SEEDS = (42, 123, 456, 789, 2026)


def load_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Results file not found: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def build_markdown_report(
    report: dict[str, Any],
) -> str:
    lines = [
        "# Detailed Arrhythmia Confusion Analysis",
        "",
        "## Aggregate test performance",
        "",
    ]

    overall_fraction = report[
        "aggregate_test"
    ]["overall_abnormal_to_normal_fraction"]

    overall_count = report[
        "aggregate_test"
    ]["overall_abnormal_to_normal_count"]

    abnormal_support = report[
        "aggregate_test"
    ]["overall_abnormal_support"]

    lines.extend(
        [
            f"Abnormal beats evaluated: {abnormal_support}",
            "",
            f"Abnormal beats predicted as N: {overall_count}",
            "",
            (
                "Abnormal to normal fraction: "
                f"{overall_fraction:.4f}"
            ),
            "",
            "## Per class test errors",
            "",
        ]
    )

    per_class = report["aggregate_test"]["per_class"]

    for label in DEFAULT_ONTOLOGY.labels:
        details = per_class[label]

        lines.extend(
            [
                f"### {label}",
                "",
                f"Support: {details['support']}",
                "",
                (
                    "Recall from aggregate confusion matrix: "
                    f"{details['recall_from_matrix']:.4f}"
                ),
                "",
            ]
        )

        if label != "N":
            lines.extend(
                [
                    (
                        "Predicted as N: "
                        f"{details['abnormal_to_normal_count']}"
                    ),
                    "",
                    (
                        "Abnormal to normal fraction: "
                        f"{details['abnormal_to_normal_fraction']:.4f}"
                    ),
                    "",
                ]
            )

        lines.append("Most common errors")
        lines.append("")

        for error in details["top_errors"][:3]:
            lines.append(
                
                    f"* {label} to "
                    f"{error['predicted_label']} "
                    f"{error['count']} "
                    f"({error['fraction']:.2%})"
                
            )

        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()

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

    input_dir = arguments.input_dir.resolve()

    test_matrices: list[np.ndarray] = []
    validation_matrices: list[np.ndarray] = []

    per_seed: dict[str, Any] = {}

    for seed in DEFAULT_SEEDS:
        result_path = (
            input_dir
            / f"seed_{seed}"
            / "results.json"
        )

        result = load_results(result_path)

        validation_matrix = np.asarray(
            result["validation"]["confusion_matrix"],
            dtype=np.int64,
        )

        test_matrix = np.asarray(
            result["test"]["confusion_matrix"],
            dtype=np.int64,
        )

        validation_matrices.append(validation_matrix)
        test_matrices.append(test_matrix)

        per_seed[str(seed)] = {
            "validation": analyse_confusion_matrix(
                validation_matrix,
                DEFAULT_ONTOLOGY.labels,
            ),
            "test": analyse_confusion_matrix(
                test_matrix,
                DEFAULT_ONTOLOGY.labels,
            ),
        }

    aggregate_validation = aggregate_confusion_matrices(
        validation_matrices
    )

    aggregate_test = aggregate_confusion_matrices(
        test_matrices
    )

    report = {
        "experiment": "repeated_seeds_confusion_analysis",
        "seeds": list(DEFAULT_SEEDS),
        "per_seed": per_seed,
        "aggregate_validation": analyse_confusion_matrix(
            aggregate_validation,
            DEFAULT_ONTOLOGY.labels,
        ),
        "aggregate_test": analyse_confusion_matrix(
            aggregate_test,
            DEFAULT_ONTOLOGY.labels,
        ),
    }

    output_dir = input_dir / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "confusion_analysis.json"

    json_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    markdown_path = output_dir / "confusion_analysis.md"

    markdown_path.write_text(
        build_markdown_report(report),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("Confusion analysis complete")
    print("=" * 70)

    aggregate = report["aggregate_test"]

    print(
        "Abnormal predicted as N: "
        f"{aggregate['overall_abnormal_to_normal_count']} "
        f"of {aggregate['overall_abnormal_support']} "
        f"({aggregate['overall_abnormal_to_normal_fraction']:.2%})"
    )

    print()

    for label in DEFAULT_ONTOLOGY.labels:
        if label == "N":
            continue

        details = aggregate["per_class"][label]

        top_error = details["top_errors"][0]

        print(
            f"{label} | "
            f"Recall {details['recall_from_matrix']:.4f} | "
            f"To N {details['abnormal_to_normal_fraction']:.2%} | "
            f"Top error {label} to "
            f"{top_error['predicted_label']} "
            f"{top_error['fraction']:.2%}"
        )

    print()
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")


if __name__ == "__main__":
    main()