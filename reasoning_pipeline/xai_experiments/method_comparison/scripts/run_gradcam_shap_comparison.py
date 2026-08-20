from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import neurokit2 as nk
import numpy as np
import wfdb

from reasoning_pipeline.baseline_adapter.classifier import BaselineClassifier
from reasoning_pipeline.infrastructure.explainability.grad_cam_1d import GradCAM1D
from xai_experiments.method_comparison.evaluation.compare_gradcam_shap import (
    compare_gradcam_shap,
)
from xai_experiments.method_comparison.explainers.shap_1d import SHAP1D


RECORD_PATH = (
    Path.home()
    / "Desktop/ecg-test-data/mitbih-119/119"
)

CHECKPOINT_PATH = (
    Path.home()
    / "Desktop/Ai-agent-for-Federated-Learning"
    / "fl_ecg_orchestrator/outputs/checkpoints"
    / "fedavg_mu_0.0_smote_0_seed_42_final_round_10.pth"
)

OUTPUT_PATH = Path(
    "xai_experiments/method_comparison/outputs/"
    "record119_beat1_gradcam_shap_metrics.json"
)

LEAD_NAME = "MLII"
BEAT_INDEX = 1
BACKGROUND_SIZE = 50
INPUT_LENGTH = 216
PRE_R_SAMPLES = 72
POST_R_SAMPLES = 144


def prepare_beat(
    cleaned_signal: np.ndarray,
    r_peak: int,
) -> np.ndarray | None:

    start = int(r_peak) - PRE_R_SAMPLES
    stop = int(r_peak) + POST_R_SAMPLES

    if start < 0 or stop > len(cleaned_signal):
        return None

    beat = cleaned_signal[start:stop].astype(
        np.float32
    )

    if beat.shape != (INPUT_LENGTH,):
        return None

    mean = float(np.mean(beat))
    std = float(np.std(beat))

    if std <= 0:
        return None

    beat = (beat - mean) / std

    return beat.astype(np.float32)


def main() -> None:

    print("\n=== Loading MIT-BIH Record 119 ===")

    record = wfdb.rdrecord(
        str(RECORD_PATH)
    )

    if LEAD_NAME not in record.sig_name:
        raise RuntimeError(
            f"Lead {LEAD_NAME} not found. "
            f"Available leads: {record.sig_name}"
        )

    lead_index = record.sig_name.index(
        LEAD_NAME
    )

    signal = np.asarray(
        record.p_signal[:, lead_index],
        dtype=np.float64,
    )

    sampling_rate = float(record.fs)

    print("Lead:", LEAD_NAME)
    print("Sampling rate:", sampling_rate)
    print("Signal samples:", len(signal))
    print(
        "Duration minutes:",
        len(signal) / sampling_rate / 60.0,
    )

    print("\n=== Preprocessing ===")

    cleaned = nk.ecg_clean(
        signal,
        sampling_rate=sampling_rate,
        method="neurokit",
    )

    _, peak_info = nk.ecg_peaks(
        cleaned,
        sampling_rate=sampling_rate,
        method="neurokit",
    )

    r_peaks = np.asarray(
        peak_info["ECG_R_Peaks"],
        dtype=np.int64,
    )

    print("Detected R peaks:", len(r_peaks))

    prepared_beats: list[np.ndarray] = []
    prepared_peak_indices: list[int] = []

    for r_peak in r_peaks:
        beat = prepare_beat(
            cleaned,
            int(r_peak),
        )

        if beat is not None:
            prepared_beats.append(beat)
            prepared_peak_indices.append(
                int(r_peak)
            )

    print(
        "Valid 216 sample beats:",
        len(prepared_beats),
    )

    if len(prepared_beats) <= BEAT_INDEX:
        raise RuntimeError(
            "Requested beat index does not exist"
        )

    target_beat = prepared_beats[
        BEAT_INDEX
    ]

    target_r_peak = prepared_peak_indices[
        BEAT_INDEX
    ]

    background_size = min(
        BACKGROUND_SIZE,
        len(prepared_beats),
    )

    background = np.asarray(
        prepared_beats[:background_size],
        dtype=np.float32,
    )

    print(
        "SHAP background beats:",
        len(background),
    )

    print("\n=== Loading frozen CNN ===")

    classifier = BaselineClassifier(
        checkpoint_path=CHECKPOINT_PATH,
        device="cpu",
    )

    prediction = classifier.predict(
        target_beat
    )

    print("Beat index:", BEAT_INDEX)
    print("R peak sample:", target_r_peak)
    print(
        "R peak time:",
        f"{target_r_peak / sampling_rate:.4f} s",
    )
    print(
        "Prediction:",
        prediction.predicted_label,
    )
    print(
        "Confidence:",
        f"{prediction.confidence * 100:.2f}%",
    )

    print("\n=== Running GradCAM ===")

    gradcam = GradCAM1D(
        model=classifier.model,
        target_layer=classifier.model.features[8],
        target_layer_name="features.8",
    )

    gradcam_result = gradcam.explain(
        samples=tuple(
            float(value)
            for value in target_beat
        ),
        target_class=prediction.predicted_class,
    )

    gradcam_values = np.asarray(
        gradcam_result.values,
        dtype=np.float64,
    )

    print(
        "GradCAM values:",
        len(gradcam_values),
    )

    print("\n=== Running SHAP ===")

    shap_explainer = SHAP1D(
        model=classifier.model,
        background=background,
    )

    shap_values = shap_explainer.explain(
        samples=target_beat,
        target_class=prediction.predicted_class,
    ).astype(np.float64)

    print(
        "SHAP values:",
        len(shap_values),
    )

    print(
        "SHAP signed range:",
        f"{shap_values.min():.6f}",
        "to",
        f"{shap_values.max():.6f}",
    )

    print(
        "\n=== Common 54 region comparison ==="
    )

    comparison = compare_gradcam_shap(
        gradcam_values=gradcam_values,
        shap_values=shap_values,
        sampling_rate_hz=sampling_rate,
    )

    print(
        "Pearson:",
        f"{comparison.pearson:.4f}",
    )
    print(
        "Spearman:",
        f"{comparison.spearman:.4f}",
    )

    print(
        "Top 10% overlap:",
        f"{comparison.top_10_overlap:.4f}",
    )
    print(
        "Top 10% IoU:",
        f"{comparison.top_10_iou:.4f}",
    )

    print(
        "Top 20% overlap:",
        f"{comparison.top_20_overlap:.4f}",
    )
    print(
        "Top 20% IoU:",
        f"{comparison.top_20_iou:.4f}",
    )

    print(
        "GradCAM peak region:",
        comparison.gradcam_peak_region,
    )
    print(
        "SHAP peak region:",
        comparison.shap_peak_region,
    )

    print(
        "Peak distance regions:",
        comparison.peak_distance_regions,
    )
    print(
        "Peak distance samples:",
        comparison.peak_distance_samples,
    )
    print(
        "Peak distance ms:",
        f"{comparison.peak_distance_ms:.2f}",
    )

    result = {
        "experiment": {
            "name": "gradcam_vs_shap",
            "common_resolution_regions": 54,
            "samples_per_region": 4,
        },
        "record": {
            "dataset": "MIT-BIH Arrhythmia Database",
            "record": "119",
            "lead": LEAD_NAME,
            "sampling_rate_hz": sampling_rate,
            "signal_samples": len(signal),
            "duration_minutes": (
                len(signal)
                / sampling_rate
                / 60.0
            ),
            "detected_r_peaks": len(r_peaks),
            "valid_beats": len(prepared_beats),
        },
        "beat": {
            "beat_index": BEAT_INDEX,
            "r_peak_sample": target_r_peak,
            "r_peak_time_seconds": (
                target_r_peak
                / sampling_rate
            ),
            "input_length": INPUT_LENGTH,
        },
        "prediction": {
            "class_index": (
                prediction.predicted_class
            ),
            "label": (
                prediction.predicted_label
            ),
            "confidence": (
                prediction.confidence
            ),
            "model_version": (
                prediction.model_version
            ),
            "checkpoint_hash": (
                prediction.checkpoint_hash
            ),
        },
        "shap": {
            "background_size": (
                background_size
            ),
            "minimum": float(
                shap_values.min()
            ),
            "maximum": float(
                shap_values.max()
            ),
            "mean_absolute_attribution": float(
                np.mean(
                    np.abs(shap_values)
                )
            ),
        },
        "comparison": asdict(
            comparison
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    print("\n=== Experiment saved ===")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()