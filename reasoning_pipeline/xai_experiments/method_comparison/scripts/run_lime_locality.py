from __future__ import annotations

import json
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

OUTPUT_PATH = Path(
    "xai_experiments/method_comparison/outputs/"
    "record119_beat1_lime_fine_locality.json"
)

SEEDS = (42, 43, 44, 45, 46)

MASK_PROBABILITIES = (
    0.01,
)

NUM_SAMPLES = 2500
BEAT_INDEX = 1


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
        int(round(
            len(importance) * fraction
        )),
    )

    return set(
        np.argsort(
            importance
        )[-k:].tolist()
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

    return (
        len(first_top & second_top)
        / len(first_top)
    )


def stability(
    explanations: dict[int, np.ndarray],
) -> dict[str, float]:

    pearsons = []
    spearmans = []
    top10 = []
    top20 = []

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

        top10.append(
            overlap(
                first,
                second,
                0.10,
            )
        )

        top20.append(
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
        "mean_spearman": float(
            np.mean(spearmans)
        ),
        "mean_top10": float(
            np.mean(top10)
        ),
        "mean_top20": float(
            np.mean(top20)
        ),
    }


def main() -> None:

    print(
        "\n=== LIME locality experiment ==="
    )

    record = wfdb.rdrecord(
        str(RECORD_PATH)
    )

    lead_index = record.sig_name.index(
        "MLII"
    )

    signal = np.asarray(
        record.p_signal[:, lead_index],
        dtype=np.float64,
    )

    fs = float(record.fs)

    cleaned = nk.ecg_clean(
        signal,
        sampling_rate=fs,
        method="neurokit",
    )

    _, information = nk.ecg_peaks(
        cleaned,
        sampling_rate=fs,
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
        "Prediction:",
        prediction.predicted_label,
    )

    print(
        "Confidence:",
        f"{prediction.confidence * 100:.2f}%",
    )

    print(
        "Perturbations per run:",
        NUM_SAMPLES,
    )

    results = []

    for mask_probability in MASK_PROBABILITIES:

        print(
            "\n================================"
        )

        print(
            "Mask probability:",
            f"{mask_probability:.0%}",
        )

        print(
            "================================"
        )

        explanations = {}

        strongest_regions = []

        retained_target_probabilities = []

        retained_target_classes = []

        for seed in SEEDS:

            lime = LIME1D(
                model=classifier.model,
                num_samples=NUM_SAMPLES,
                random_state=seed,
                masking_value=0.0,
                mask_probability=mask_probability,
            )

            values = lime.explain(
                samples=target,
                target_class=(
                    prediction.predicted_class
                ),
            )

            explanations[
                seed
            ] = values

            strongest = int(
                np.argmax(
                    np.abs(values)
                )
            )

            strongest_regions.append(
                strongest
            )

            masks = lime._generate_masks()

            perturbed = lime._apply_masks(
                target,
                masks,
            )

            probabilities = (
                lime._predict_proba(
                    perturbed
                )
            )

            target_probabilities = (
                probabilities[
                    :,
                    prediction.predicted_class,
                ]
            )

            predicted_classes = np.argmax(
                probabilities,
                axis=1,
            )

            mean_target_probability = float(
                np.mean(
                    target_probabilities
                )
            )

            same_class_fraction = float(
                np.mean(
                    predicted_classes
                    == prediction.predicted_class
                )
            )

            retained_target_probabilities.append(
                mean_target_probability
            )

            retained_target_classes.append(
                same_class_fraction
            )

            print(
                f"Seed {seed}: "
                f"strongest={strongest}, "
                f"mean target prob="
                f"{mean_target_probability:.4f}, "
                f"same class="
                f"{same_class_fraction:.4f}"
            )

        metrics = stability(
            explanations
        )

        mean_target_probability = float(
            np.mean(
                retained_target_probabilities
            )
        )

        mean_same_class = float(
            np.mean(
                retained_target_classes
            )
        )

        print("\nStability")

        print(
            "Mean Pearson:",
            f"{metrics['mean_pearson']:.4f}",
        )

        print(
            "Mean Spearman:",
            f"{metrics['mean_spearman']:.4f}",
        )

        print(
            "Mean top 10% overlap:",
            f"{metrics['mean_top10']:.4f}",
        )

        print(
            "Mean top 20% overlap:",
            f"{metrics['mean_top20']:.4f}",
        )

        print("\nLocality")

        print(
            "Mean target probability:",
            f"{mean_target_probability:.4f}",
        )

        print(
            "Fraction retaining target class:",
            f"{mean_same_class:.4f}",
        )

        print(
            "Strongest regions:",
            strongest_regions,
        )

        results.append(
            {
                "mask_probability": (
                    mask_probability
                ),
                "strongest_regions": (
                    strongest_regions
                ),
                "mean_target_probability": (
                    mean_target_probability
                ),
                "fraction_retaining_target_class": (
                    mean_same_class
                ),
                **metrics,
            }
        )

    print(
        "\n\n=== Locality summary ==="
    )

    print(
        "Mask | Pearson | Spearman | "
        "Top10 | Top20 | TargetProb | SameClass"
    )

    print(
        "--------------------------------"
        "--------------------------------"
    )

    for result in results:

        print(
            f"{result['mask_probability']:4.0%} | "
            f"{result['mean_pearson']:7.4f} | "
            f"{result['mean_spearman']:8.4f} | "
            f"{result['mean_top10']:5.3f} | "
            f"{result['mean_top20']:5.3f} | "
            f"{result['mean_target_probability']:10.4f} | "
            f"{result['fraction_retaining_target_class']:9.4f}"
        )

    output = {
        "experiment": "lime_locality",
        "record": "119",
        "beat_index": BEAT_INDEX,
        "prediction": (
            prediction.predicted_label
        ),
        "original_confidence": float(
            prediction.confidence
        ),
        "num_samples": NUM_SAMPLES,
        "seeds": list(SEEDS),
        "results": results,
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