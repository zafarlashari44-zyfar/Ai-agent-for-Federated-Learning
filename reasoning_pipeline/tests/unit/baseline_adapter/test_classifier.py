from pathlib import Path

import numpy as np
import pytest

from reasoning_pipeline.baseline_adapter import (
    BaselineClassifier,
    InvalidBeatError,
)

CHECKPOINT = Path(
    "../fl_ecg_orchestrator/outputs/checkpoints/"
    "fedavg_mu_0.0_smote_0_seed_42_final_round_10.pth"
).resolve()


@pytest.fixture(scope="module")
def classifier() -> BaselineClassifier:
    return BaselineClassifier(
        checkpoint_path=CHECKPOINT,
        device="cpu",
    )


def test_checkpoint_exists() -> None:
    assert CHECKPOINT.is_file()


def test_prediction_is_valid(
    classifier: BaselineClassifier,
) -> None:
    beat = np.zeros(
        216,
        dtype=np.float32,
    )

    prediction = classifier.predict(beat)

    assert prediction.predicted_class in range(5)
    assert prediction.predicted_label in {
        "N",
        "S",
        "V",
        "F",
        "Q",
    }
    assert len(prediction.probabilities) == 5
    assert sum(prediction.probabilities) == pytest.approx(
        1.0,
        abs=1e-6,
    )
    assert prediction.confidence == pytest.approx(
        prediction.probabilities[
            prediction.predicted_class
        ]
    )


def test_prediction_has_reproducibility_metadata(
    classifier: BaselineClassifier,
) -> None:
    prediction = classifier.predict(
        np.zeros(216, dtype=np.float32)
    )

    assert prediction.checkpoint_path == str(CHECKPOINT)
    assert prediction.checkpoint_hash == (
        "e9119a15cf9bb8b3d15c085e7be4889"
        "a49f0649ef27682be03a979d6eba0faa2"
    )
    assert prediction.model_version == (
        "fedavg-round-10-v1"
    )
    assert prediction.preprocessing_version == (
        "scribe-v1-neurokit"
    )


def test_rejects_incorrect_length(
    classifier: BaselineClassifier,
) -> None:
    with pytest.raises(
        InvalidBeatError,
        match="Expected 216 samples",
    ):
        classifier.predict(
            np.zeros(200, dtype=np.float32)
        )


def test_rejects_non_finite_values(
    classifier: BaselineClassifier,
) -> None:
    beat = np.zeros(
        216,
        dtype=np.float32,
    )
    beat[10] = np.nan

    with pytest.raises(
        InvalidBeatError,
        match="NaN or infinite",
    ):
        classifier.predict(beat)


def test_prediction_is_deterministic(
    classifier: BaselineClassifier,
) -> None:
    beat = np.linspace(
        -1.0,
        1.0,
        216,
        dtype=np.float32,
    )

    first = classifier.predict(beat)
    second = classifier.predict(beat)

    assert first.predicted_class == second.predicted_class
    assert first.probabilities == pytest.approx(
        second.probabilities,
        abs=1e-8,
    )
