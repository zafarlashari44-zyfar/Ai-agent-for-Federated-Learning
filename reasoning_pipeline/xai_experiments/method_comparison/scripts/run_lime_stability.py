from __future__ import annotations

from itertools import combinations
from pathlib import Path

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

LEAD_NAME = "MLII"
BEAT_INDEX = 1

SEEDS = (42, 43, 44, 45, 46)
NUM_SAMPLES = 1000


def prepare_beat(
    cleaned: np.ndarray,
    peak: int,
) -> np.ndarray | None:

    start = int(peak) - 72
    stop = int(peak) + 144

    if start < 0 or stop > len(cleaned):
        return None

    beat = cleaned[start:stop].astype(np.float32)

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

    return set(
        np.argsort(importance)[-k:].tolist()
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

    return len(
        first_top & second_top
    ) / len(first_top)


def main() -> None:

    print("\n=== Loading Record 119 ===")

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

    sampling_rate = float(record.fs)

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

    target = beats[BEAT_INDEX]

    classifier = BaselineClassifier(
        checkpoint_path=CHECKPOINT_PATH,
        device="cpu",
    )

    prediction = classifier.predict(
        target
    )

    print("Beat index:", BEAT_INDEX)
    print(
        "Prediction:",
        prediction.predicted_label,
    )
    print(
        "Confidence:",
        f"{prediction.confidence * 100:.2f}%",
    )
    print("LIME samples per run:", NUM_SAMPLES)

    explanations: dict[int, np.ndarray] = {}

    print("\n=== Running LIME seeds ===")

    for seed in SEEDS:

        lime = LIME1D(
            model=classifier.model,
            num_samples=NUM_SAMPLES,
            random_state=seed,
            masking_value=0.0,
        )

        values = lime.explain(
            samples=target,
            target_class=prediction.predicted_class,
        )

        explanations[seed] = values

        strongest = int(
            np.argmax(
                np.abs(values)
            )
        )

        print(
            f"Seed {seed}: "
            f"strongest region {strongest}, "
            f"mean |weight| "
            f"{np.mean(np.abs(values)):.6f}"
        )

    print("\n=== Pairwise stability ===")

    pearsons = []
    spearmans = []
    overlaps_10 = []
    overlaps_20 = []

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

        pearson = float(
            pearsonr(
                first,
                second,
            ).statistic
        )

        spearman = float(
            spearmanr(
                first,
                second,
            ).statistic
        )

        overlap_10 = overlap(
            first,
            second,
            0.10,
        )

        overlap_20 = overlap(
            first,
            second,
            0.20,
        )

        pearsons.append(pearson)
        spearmans.append(spearman)
        overlaps_10.append(overlap_10)
        overlaps_20.append(overlap_20)

        print(
            f"{first_seed} vs {second_seed}: "
            f"Pearson={pearson:.4f}, "
            f"Spearman={spearman:.4f}, "
            f"Top10={overlap_10:.4f}, "
            f"Top20={overlap_20:.4f}"
        )

    print("\n=== Stability summary ===")

    print(
        "Mean Pearson:",
        f"{np.mean(pearsons):.4f}",
    )

    print(
        "Minimum Pearson:",
        f"{np.min(pearsons):.4f}",
    )

    print(
        "Mean Spearman:",
        f"{np.mean(spearmans):.4f}",
    )

    print(
        "Minimum Spearman:",
        f"{np.min(spearmans):.4f}",
    )

    print(
        "Mean top 10% overlap:",
        f"{np.mean(overlaps_10):.4f}",
    )

    print(
        "Mean top 20% overlap:",
        f"{np.mean(overlaps_20):.4f}",
    )


if __name__ == "__main__":
    main()