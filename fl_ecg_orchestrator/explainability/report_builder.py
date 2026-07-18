from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from fl_ecg_orchestrator.data.loader import CLASS_NAMES, resolve_project_path


CLINICAL_CLASS_NAMES = {
    0: "Normal beat",
    1: "Supraventricular ectopic beat",
    2: "Ventricular ectopic beat",
    3: "Fusion beat",
    4: "Unknown or unclassifiable beat",
}


def confidence_level(confidence: float) -> str:
    if confidence >= 0.90:
        return "High"
    if confidence >= 0.70:
        return "Moderate"
    return "Low"


def review_recommendation(prediction: int, confidence: float) -> str:
    if confidence < 0.70:
        return "Manual clinician review recommended due to low confidence."
    if prediction == 0:
        return "No automated escalation; interpret alongside the full ECG and clinical context."
    return "Clinician review recommended because an abnormal beat class was predicted."


def build_reports(
    results: dict[str, Any],
    output_dir: str | Path,
    top_features: int = 10,
) -> Path:
    output_path = resolve_project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, Any]] = []

    for position in range(len(results["features"])):
        prediction = int(results["predictions"][position])
        true_label = int(results["true_labels"][position])
        confidence = float(
            results["probabilities"][position, prediction]
        )
        values = results["shap_values"][position, prediction]
        feature_values = results["features"][position]

        order = np.argsort(np.abs(values))[::-1][:top_features]

        top_contributors = [
            {
                "feature_index": int(index),
                "feature_name": f"ECG sample {int(index):03d}",
                "feature_value": float(feature_values[index]),
                "shap_value": float(values[index]),
                "direction": (
                    "supports prediction"
                    if values[index] >= 0
                    else "opposes prediction"
                ),
            }
            for index in order
        ]

        reports.append(
            {
                "sample_position": int(position),
                "global_test_index": int(
                    results["global_indices"][position]
                ),
                "true_class_code": CLASS_NAMES[true_label],
                "true_class_name": CLINICAL_CLASS_NAMES[true_label],
                "predicted_class_code": CLASS_NAMES[prediction],
                "predicted_class_name": CLINICAL_CLASS_NAMES[prediction],
                "correct": bool(prediction == true_label),
                "confidence": confidence,
                "confidence_level": confidence_level(confidence),
                "uncertain": bool(confidence < 0.70),
                "recommendation": review_recommendation(
                    prediction,
                    confidence,
                ),
                "top_contributors": top_contributors,
            }
        )

    json_path = output_path / "individual_shap_reports.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(reports, file, indent=2)

    csv_path = output_path / "prediction_explanations.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sample_position",
                "global_test_index",
                "true_class",
                "predicted_class",
                "correct",
                "confidence",
                "confidence_level",
                "uncertain",
                "top_feature_indices",
                "top_shap_values",
                "recommendation",
            ]
        )

        for report in reports:
            writer.writerow(
                [
                    report["sample_position"],
                    report["global_test_index"],
                    report["true_class_code"],
                    report["predicted_class_code"],
                    report["correct"],
                    report["confidence"],
                    report["confidence_level"],
                    report["uncertain"],
                    "|".join(
                        str(item["feature_index"])
                        for item in report["top_contributors"]
                    ),
                    "|".join(
                        f'{item["shap_value"]:.8f}'
                        for item in report["top_contributors"]
                    ),
                    report["recommendation"],
                ]
            )

    return json_path
