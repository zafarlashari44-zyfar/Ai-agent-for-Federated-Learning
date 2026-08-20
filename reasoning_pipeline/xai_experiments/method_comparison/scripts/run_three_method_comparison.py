from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import neurokit2 as nk
import numpy as np
import torch
import wfdb
from scipy.stats import pearsonr, spearmanr

from reasoning_pipeline.baseline_adapter.classifier import BaselineClassifier
from reasoning_pipeline.infrastructure.explainability.grad_cam_1d import GradCAM1D
from xai_experiments.method_comparison.explainers.lime_1d import LIME1D
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
    "record119_three_method_comparison.json"
)

LEAD_NAME = "MLII"

NUM_BEATS = 20
BACKGROUND_SIZE = 50

NUM_REGIONS = 54
SAMPLES_PER_REGION = 4

LIME_NUM_SAMPLES = 2500
LIME_MASK_PROBABILITY = 0.01
LIME_RANDOM_STATE = 42

TOP_FRACTIONS = (
    0.10,
    0.20,
)

FAITHFULNESS_FRACTIONS = (
    0.10,
    0.20,
)


def prepare_beat(
    cleaned: np.ndarray,
    peak: int,
) -> np.ndarray | None:
    start = int(peak) - 72
    stop = int(peak) + 144

    if start < 0 or stop > len(cleaned):
        return None

    beat = cleaned[start:stop].astype(
        np.float32
    )

    if beat.shape != (216,):
        return None

    std = float(
        np.std(beat)
    )

    if std <= 0:
        return None

    beat = (
        beat - float(np.mean(beat))
    ) / std

    return beat.astype(
        np.float32
    )


def aggregate_to_regions(
    values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.shape == (NUM_REGIONS,):
        return values.copy()

    if values.shape != (216,):
        raise ValueError(
            "Attribution must contain either "
            "216 samples or 54 regions."
        )

    return values.reshape(
        NUM_REGIONS,
        SAMPLES_PER_REGION,
    ).mean(
        axis=1
    )


def normalise_importance(
    values: np.ndarray,
) -> np.ndarray:
    values = np.abs(
        np.asarray(
            values,
            dtype=np.float64,
        )
    )

    maximum = float(
        np.max(values)
    )

    if maximum > 0:
        return values / maximum

    return np.zeros_like(
        values
    )


def safe_pearson(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    if np.std(first) == 0:
        return 0.0

    if np.std(second) == 0:
        return 0.0

    return float(
        pearsonr(
            first,
            second,
        ).statistic
    )


def safe_spearman(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    if np.std(first) == 0:
        return 0.0

    if np.std(second) == 0:
        return 0.0

    return float(
        spearmanr(
            first,
            second,
        ).statistic
    )


def top_indices(
    values: np.ndarray,
    fraction: float,
) -> set[int]:
    k = max(
        1,
        int(
            round(
                len(values)
                * fraction
            )
        ),
    )

    return set(
        np.argsort(
            values
        )[-k:].tolist()
    )


def overlap_metrics(
    first: np.ndarray,
    second: np.ndarray,
    fraction: float,
) -> tuple[float, float]:
    first_top = top_indices(
        first,
        fraction,
    )

    second_top = top_indices(
        second,
        fraction,
    )

    intersection = len(
        first_top & second_top
    )

    union = len(
        first_top | second_top
    )

    overlap = (
        intersection
        / len(first_top)
    )

    iou = (
        intersection / union
        if union
        else 0.0
    )

    return (
        float(overlap),
        float(iou),
    )


def compare_pair(
    first: np.ndarray,
    second: np.ndarray,
) -> dict[str, float]:
    result = {
        "pearson": safe_pearson(
            first,
            second,
        ),
        "spearman": safe_spearman(
            first,
            second,
        ),
    }

    for fraction in TOP_FRACTIONS:
        overlap, iou = overlap_metrics(
            first,
            second,
            fraction,
        )

        percentage = int(
            fraction * 100
        )

        result[
            f"top_{percentage}_overlap"
        ] = overlap

        result[
            f"top_{percentage}_iou"
        ] = iou

    first_peak = int(
        np.argmax(first)
    )

    second_peak = int(
        np.argmax(second)
    )

    result[
        "peak_distance_regions"
    ] = abs(
        first_peak
        - second_peak
    )

    return result


def mask_top_regions(
    beat: np.ndarray,
    importance: np.ndarray,
    fraction: float,
) -> np.ndarray:
    masked = beat.copy()

    important_regions = top_indices(
        importance,
        fraction,
    )

    for region in important_regions:
        start = (
            region
            * SAMPLES_PER_REGION
        )

        stop = (
            start
            + SAMPLES_PER_REGION
        )

        masked[
            start:stop
        ] = 0.0

    return masked


def target_probability(
    classifier: BaselineClassifier,
    beat: np.ndarray,
    target_class: int,
) -> float:
    classifier.model.eval()

    tensor = torch.tensor(
        beat,
        dtype=torch.float32,
    ).unsqueeze(0)

    with torch.no_grad():
        logits = classifier.model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

    return float(
        probabilities[
            0,
            target_class,
        ].item()
    )


def calculate_faithfulness(
    classifier: BaselineClassifier,
    beat: np.ndarray,
    importance: np.ndarray,
    target_class: int,
    original_probability: float,
) -> dict[str, float]:
    result = {}

    for fraction in FAITHFULNESS_FRACTIONS:
        masked = mask_top_regions(
            beat,
            importance,
            fraction,
        )

        masked_probability = (
            target_probability(
                classifier,
                masked,
                target_class,
            )
        )

        probability_drop = (
            original_probability
            - masked_probability
        )

        percentage = int(
            fraction * 100
        )

        result[
            f"remove_top_{percentage}_probability"
        ] = masked_probability

        result[
            f"remove_top_{percentage}_drop"
        ] = probability_drop

    return result


def mean_metrics(
    records: list[dict[str, float]],
) -> dict[str, float]:
    if not records:
        return {}

    keys = records[0].keys()

    return {
        key: float(
            np.mean(
                [
                    record[key]
                    for record in records
                ]
            )
        )
        for key in keys
    }


def main() -> None:
    print(
        "\n=== Three method XAI comparison ==="
    )

    print(
        "Loading MIT-BIH Record 119"
    )

    record = wfdb.rdrecord(
        str(RECORD_PATH)
    )

    lead_index = (
        record.sig_name.index(
            LEAD_NAME
        )
    )

    signal = np.asarray(
        record.p_signal[
            :,
            lead_index,
        ],
        dtype=np.float64,
    )

    sampling_rate = float(
        record.fs
    )

    cleaned = nk.ecg_clean(
        signal,
        sampling_rate=sampling_rate,
        method="neurokit",
    )

    _, information = nk.ecg_peaks(
        cleaned,
        sampling_rate=sampling_rate,
        method="neurokit",
    )

    peaks = np.asarray(
        information[
            "ECG_R_Peaks"
        ],
        dtype=np.int64,
    )

    prepared = []

    for peak in peaks:
        beat = prepare_beat(
            cleaned,
            int(peak),
        )

        if beat is not None:
            prepared.append(
                (
                    int(peak),
                    beat,
                )
            )

    print(
        "Valid beats:",
        len(prepared),
    )

    if len(prepared) < NUM_BEATS:
        raise RuntimeError(
            "Not enough valid beats."
        )

    # Deterministic coverage across the recording
    selected_indices = np.linspace(
        0,
        len(prepared) - 1,
        NUM_BEATS,
        dtype=int,
    )

    # Keep the SHAP reference fixed for every beat.
    background = np.asarray(
        [
            beat
            for _, beat
            in prepared[
                :BACKGROUND_SIZE
            ]
        ],
        dtype=np.float32,
    )

    classifier = BaselineClassifier(
        checkpoint_path=CHECKPOINT_PATH,
        device="cpu",
    )

    gradcam = GradCAM1D(
        model=classifier.model,
        target_layer=(
            classifier.model.features[8]
        ),
        target_layer_name="features.8",
    )

    shap_explainer = SHAP1D(
        model=classifier.model,
        background=background,
    )

    lime = LIME1D(
        model=classifier.model,
        num_samples=LIME_NUM_SAMPLES,
        random_state=LIME_RANDOM_STATE,
        masking_value=0.0,
        mask_probability=(
            LIME_MASK_PROBABILITY
        ),
    )

    beat_results = []

    pairwise_records = {
        "gradcam_vs_shap": [],
        "gradcam_vs_lime": [],
        "shap_vs_lime": [],
    }

    faithfulness_records = {
        "gradcam": [],
        "shap": [],
        "lime": [],
    }

    runtime_records = {
        "gradcam": [],
        "shap": [],
        "lime": [],
    }

    print(
        "\nSelected beats:",
        selected_indices.tolist(),
    )

    for position, beat_index in enumerate(
        selected_indices,
        start=1,
    ):
        peak, beat = prepared[
            int(beat_index)
        ]

        prediction = classifier.predict(
            beat
        )

        target_class = (
            prediction.predicted_class
        )

        original_probability = float(
            prediction.confidence
        )

        print(
            "\n--------------------------------"
        )

        print(
            f"Beat {position}/{NUM_BEATS}"
        )

        print(
            "Beat index:",
            int(beat_index),
        )

        print(
            "Time:",
            f"{peak / sampling_rate:.2f}s",
        )

        print(
            "Prediction:",
            prediction.predicted_label,
        )

        print(
            "Confidence:",
            f"{original_probability * 100:.2f}%",
        )

        # GradCAM

        started = time.perf_counter()

        gradcam_result = gradcam.explain(
            samples=tuple(
                float(value)
                for value in beat
            ),
            target_class=target_class,
        )

        gradcam_runtime = (
            time.perf_counter()
            - started
        )

        gradcam_regions = (
            aggregate_to_regions(
                np.asarray(
                    gradcam_result.values,
                    dtype=np.float64,
                )
            )
        )

        gradcam_importance = (
            normalise_importance(
                gradcam_regions
            )
        )

        # SHAP

        started = time.perf_counter()

        shap_values = shap_explainer.explain(
            samples=beat,
            target_class=target_class,
        )

        shap_runtime = (
            time.perf_counter()
            - started
        )

        shap_regions = (
            aggregate_to_regions(
                np.asarray(
                    shap_values,
                    dtype=np.float64,
                )
            )
        )

        shap_importance = (
            normalise_importance(
                shap_regions
            )
        )

        # LIME

        started = time.perf_counter()

        lime_values = lime.explain(
            samples=beat,
            target_class=target_class,
        )

        lime_runtime = (
            time.perf_counter()
            - started
        )

        lime_regions = (
            aggregate_to_regions(
                np.asarray(
                    lime_values,
                    dtype=np.float64,
                )
            )
        )

        lime_importance = (
            normalise_importance(
                lime_regions
            )
        )

        methods = {
            "gradcam": gradcam_importance,
            "shap": shap_importance,
            "lime": lime_importance,
        }

        runtimes = {
            "gradcam": gradcam_runtime,
            "shap": shap_runtime,
            "lime": lime_runtime,
        }

        # Pairwise agreement

        beat_pairwise = {}

        for first_name, second_name in (
            ("gradcam", "shap"),
            ("gradcam", "lime"),
            ("shap", "lime"),
        ):
            key = (
                f"{first_name}_vs_"
                f"{second_name}"
            )

            comparison = compare_pair(
                methods[first_name],
                methods[second_name],
            )

            beat_pairwise[
                key
            ] = comparison

            pairwise_records[
                key
            ].append(
                comparison
            )

        # Faithfulness

        beat_faithfulness = {}

        for method_name, importance in (
            methods.items()
        ):
            faithfulness = (
                calculate_faithfulness(
                    classifier,
                    beat,
                    importance,
                    target_class,
                    original_probability,
                )
            )

            beat_faithfulness[
                method_name
            ] = faithfulness

            faithfulness_records[
                method_name
            ].append(
                faithfulness
            )

            runtime_records[
                method_name
            ].append(
                runtimes[
                    method_name
                ]
            )

        print(
            "Runtime "
            f"G={gradcam_runtime:.4f}s "
            f"S={shap_runtime:.4f}s "
            f"L={lime_runtime:.4f}s"
        )

        print(
            "Top 10 probability drops "
            f"G={beat_faithfulness['gradcam']['remove_top_10_drop']:.4f} "
            f"S={beat_faithfulness['shap']['remove_top_10_drop']:.4f} "
            f"L={beat_faithfulness['lime']['remove_top_10_drop']:.4f}"
        )

        beat_results.append(
            {
                "beat_index": int(
                    beat_index
                ),
                "r_peak_sample": int(
                    peak
                ),
                "r_peak_time_seconds": float(
                    peak / sampling_rate
                ),
                "predicted_class": int(
                    target_class
                ),
                "predicted_label": (
                    prediction.predicted_label
                ),
                "confidence": (
                    original_probability
                ),
                "runtime_seconds": (
                    runtimes
                ),
                "pairwise_agreement": (
                    beat_pairwise
                ),
                "faithfulness": (
                    beat_faithfulness
                ),
                "peak_regions": {
                    name: int(
                        np.argmax(
                            importance
                        )
                    )
                    for name, importance
                    in methods.items()
                },
            }
        )

    pairwise_summary = {
        key: mean_metrics(
            values
        )
        for key, values
        in pairwise_records.items()
    }

    faithfulness_summary = {
        key: mean_metrics(
            values
        )
        for key, values
        in faithfulness_records.items()
    }

    runtime_summary = {
        key: float(
            np.mean(values)
        )
        for key, values
        in runtime_records.items()
    }

    print(
        "\n\n=== Pairwise agreement summary ==="
    )

    for name, metrics in (
        pairwise_summary.items()
    ):
        print(
            f"\n{name}"
        )

        print(
            "Pearson:",
            f"{metrics['pearson']:.4f}",
        )

        print(
            "Spearman:",
            f"{metrics['spearman']:.4f}",
        )

        print(
            "Top10 overlap:",
            f"{metrics['top_10_overlap']:.4f}",
        )

        print(
            "Top20 overlap:",
            f"{metrics['top_20_overlap']:.4f}",
        )

    print(
        "\n=== Faithfulness summary ==="
    )

    for name, metrics in (
        faithfulness_summary.items()
    ):
        print(
            f"\n{name}"
        )

        print(
            "Top10 probability drop:",
            f"{metrics['remove_top_10_drop']:.4f}",
        )

        print(
            "Top20 probability drop:",
            f"{metrics['remove_top_20_drop']:.4f}",
        )

    print(
        "\n=== Runtime summary ==="
    )

    for name, runtime in (
        runtime_summary.items()
    ):
        print(
            name,
            f"{runtime:.4f}s",
        )

    output = {
        "experiment": (
            "gradcam_shap_lime_comparison"
        ),
        "record": "119",
        "lead": LEAD_NAME,
        "sampling_rate_hz": (
            sampling_rate
        ),
        "num_valid_beats": len(
            prepared
        ),
        "num_evaluated_beats": (
            NUM_BEATS
        ),
        "selected_beat_indices": (
            selected_indices.tolist()
        ),
        "common_regions": (
            NUM_REGIONS
        ),
        "lime_configuration": {
            "num_samples": (
                LIME_NUM_SAMPLES
            ),
            "mask_probability": (
                LIME_MASK_PROBABILITY
            ),
            "random_state": (
                LIME_RANDOM_STATE
            ),
            "guaranteed_nonempty_perturbation": True,
        },
        "pairwise_summary": (
            pairwise_summary
        ),
        "faithfulness_summary": (
            faithfulness_summary
        ),
        "runtime_summary_seconds": (
            runtime_summary
        ),
        "beats": beat_results,
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
            output,
            file,
            indent=2,
        )

    print(
        "\nResults saved to"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()