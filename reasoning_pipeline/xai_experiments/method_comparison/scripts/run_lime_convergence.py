from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from time import perf_counter

import neurokit2 as nk
import numpy as np
import wfdb
from scipy.stats import pearsonr, spearmanr

from reasoning_pipeline.baseline_adapter.classifier import BaselineClassifier
from xai_experiments.method_comparison.explainers.lime_1d import LIME1D


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
    "record119_beat1_lime_convergence.json"
)

LEAD_NAME = "MLII"
BEAT_INDEX = 1

SEEDS = (42, 43, 44, 45, 46)

SAMPLE_COUNTS = (
    1000,
    2500,
    5000,
    10000,
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

    std = float(np.std(beat))

    if std <= 0:
        return None

    return (
        (beat - float(np.mean(beat))) / std
    ).astype(np.float32)


def top_regions(
    values: np.ndarray,
    fraction: float,
) -> set[int]:

    importance = np.abs(values)

    k = max(
        1,
        int(round(len(importance) * fraction)),
    )

    indices = np.argsort(
        importance
    )[-k:]

    return set(
        int(index)
        for index in indices
    )


def overlap(
    first: np.ndarray,
    second: np.ndarray,
    fraction: float,
) -> float:

    first_top = top_regions(
        first,
        fraction,
    )

    second_top = top_regions(
        second,
        fraction,
    )

    intersection = len(
        first_top & second_top
    )

    return (
        intersection / len(first_top)
        if first_top
        else 0.0
    )


def calculate_stability(
    explanations: dict[int, np.ndarray],
) -> dict[str, float]:

    pearsons = []
    spearmans = []
    top_10_overlaps = []
    top_20_overlaps = []

    for first_seed, second_seed in combinations(
        SEEDS,
        2,
    ):

        first = np.abs(
            explanations[first_seed]
        )

        second = np.abs(
            explanations[second_seed]
        )

        pearsons.append(
            float(
                pearsonr(
                    first,
                    second,
                ).statistic
            )
        )

        spearmans.append(
            float(
                spearmanr(
                    first,
                    second,
                ).statistic
            )
        )

        top_10_overlaps.append(
            overlap(
                first,
                second,
                0.10,
            )
        )

        top_20_overlaps.append(
            overlap(
                first,
                second,
                0.20,
            )
        )

    return {
        "mean_pearson": float(
            np.mean(pearsons)
        ),
        "minimum_pearson": float(
            np.min(pearsons)
        ),
        "mean_spearman": float(
            np.mean(spearmans)
        ),
        "minimum_spearman": float(
            np.min(spearmans)
        ),
        "mean_top_10_overlap": float(
            np.mean(top_10_overlaps)
        ),
        "mean_top_20_overlap": float(
            np.mean(top_20_overlaps)
        ),
    }


def main() -> None:

    print(
        "\n=== LIME convergence experiment ==="
    )

    print("\nLoading MIT-BIH Record 119")

    record = wfdb.rdrecord(
        str(RECORD_PATH)
    )

    lead_index = record.sig_name.index(
        LEAD_NAME
    )

    signal = np.asarray(
        record.p_signal[:, lead_index],
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
        information["ECG_R_Peaks"],
        dtype=np.int64,
    )

    beats = []

    for peak in peaks:

        beat = prepare_beat(
            cleaned,
            int(peak),
        )

        if beat is not None:
            beats.append(beat)

    if len(beats) <= BEAT_INDEX:
        raise RuntimeError(
            "Requested beat does not exist"
        )

    target = beats[
        BEAT_INDEX
    ]

    classifier = BaselineClassifier(
        checkpoint_path=CHECKPOINT_PATH,
        device="cpu",
    )

    prediction = classifier.predict(
        target
    )

    print(
        "Beat index:",
        BEAT_INDEX,
    )

    print(
        "Prediction:",
        prediction.predicted_label,
    )

    print(
        "Confidence:",
        f"{prediction.confidence * 100:.2f}%",
    )

    print(
        "Sample counts:",
        SAMPLE_COUNTS,
    )

    print(
        "Seeds:",
        SEEDS,
    )

    experiment_results = []

    for num_samples in SAMPLE_COUNTS:

        print(
            "\n========================================"
        )

        print(
            "LIME perturbations:",
            num_samples,
        )

        print(
            "========================================"
        )

        explanations = {}

        run_times = []

        strongest_regions = []

        for seed in SEEDS:

            lime = LIME1D(
                model=classifier.model,
                num_samples=num_samples,
                random_state=seed,
                masking_value=0.0,
            )

            start_time = perf_counter()

            values = lime.explain(
                samples=target,
                target_class=prediction.predicted_class,
            )

            elapsed = (
                perf_counter()
                - start_time
            )

            explanations[
                seed
            ] = values

            run_times.append(
                elapsed
            )

            strongest_region = int(
                np.argmax(
                    np.abs(values)
                )
            )

            strongest_regions.append(
                strongest_region
            )

            print(
                f"Seed {seed}: "
                f"strongest={strongest_region}, "
                f"mean|weight|="
                f"{np.mean(np.abs(values)):.6f}, "
                f"time={elapsed:.2f}s"
            )

        stability = calculate_stability(
            explanations
        )

        mean_runtime = float(
            np.mean(run_times)
        )

        print(
            "\nStability"
        )

        print(
            "Mean Pearson:",
            f"{stability['mean_pearson']:.4f}",
        )

        print(
            "Minimum Pearson:",
            f"{stability['minimum_pearson']:.4f}",
        )

        print(
            "Mean Spearman:",
            f"{stability['mean_spearman']:.4f}",
        )

        print(
            "Minimum Spearman:",
            f"{stability['minimum_spearman']:.4f}",
        )

        print(
            "Mean top 10% overlap:",
            f"{stability['mean_top_10_overlap']:.4f}",
        )

        print(
            "Mean top 20% overlap:",
            f"{stability['mean_top_20_overlap']:.4f}",
        )

        print(
            "Mean runtime:",
            f"{mean_runtime:.2f}s",
        )

        print(
            "Strongest regions:",
            strongest_regions,
        )

        experiment_results.append(
            {
                "num_samples": (
                    num_samples
                ),
                "seeds": list(
                    SEEDS
                ),
                "strongest_regions": (
                    strongest_regions
                ),
                "mean_runtime_seconds": (
                    mean_runtime
                ),
                **stability,
            }
        )

    print(
        "\n\n=== Convergence summary ==="
    )

    print(
        "\nSamples | Pearson | Spearman | "
        "Top10 | Top20 | Runtime"
    )

    print(
        "--------------------------------"
        "-----------------------"
    )

    for result in experiment_results:

        print(
            f"{result['num_samples']:7d} | "
            f"{result['mean_pearson']:7.4f} | "
            f"{result['mean_spearman']:8.4f} | "
            f"{result['mean_top_10_overlap']:5.3f} | "
            f"{result['mean_top_20_overlap']:5.3f} | "
            f"{result['mean_runtime_seconds']:6.2f}s"
        )

    output = {
        "experiment": (
            "lime_sample_convergence"
        ),
        "dataset": (
            "MIT-BIH Arrhythmia Database"
        ),
        "record": "119",
        "lead": LEAD_NAME,
        "beat_index": BEAT_INDEX,
        "prediction": (
            prediction.predicted_label
        ),
        "confidence": float(
            prediction.confidence
        ),
        "masking_value": 0.0,
        "num_regions": 54,
        "results": experiment_results,
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