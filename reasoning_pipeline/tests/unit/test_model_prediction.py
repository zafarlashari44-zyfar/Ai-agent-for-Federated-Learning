from __future__ import annotations

import pytest

from reasoning_pipeline.domain.models import ModelPrediction


def create_valid_prediction() -> ModelPrediction:
    return ModelPrediction(
        predicted_class=2,
        predicted_label="Ventricular Arrhythmia",
        probabilities=(0.02, 0.03, 0.90, 0.03, 0.02),
        confidence=0.90,
        checkpoint_path="checkpoints/global_model.pt",
        checkpoint_hash="test_hash",
        model_version="baseline_1.0",
        preprocessing_version="scribe_v1",
    )


def test_model_prediction_accepts_valid_probabilities() -> None:
    prediction = create_valid_prediction()

    assert prediction.predicted_class == 2
    assert prediction.predicted_label == "Ventricular Arrhythmia"
    assert prediction.confidence == pytest.approx(0.90)
    assert sum(prediction.probabilities) == pytest.approx(1.0)


def test_model_prediction_rejects_probabilities_not_summing_to_one() -> None:
    with pytest.raises(ValueError, match="probabilities must sum to one"):
        ModelPrediction(
            predicted_class=2,
            predicted_label="Ventricular Arrhythmia",
            probabilities=(0.1, 0.1, 0.1, 0.1, 0.1),
            confidence=0.5,
            checkpoint_path="checkpoint.pt",
            checkpoint_hash="hash",
            model_version="1.0",
            preprocessing_version="1.0",
        )


def test_model_prediction_rejects_out_of_range_probability() -> None:
    with pytest.raises(
        ValueError,
        match="probabilities must be between zero and one",
    ):
        ModelPrediction(
            predicted_class=0,
            predicted_label="Normal",
            probabilities=(1.1, -0.1),
            confidence=1.0,
            checkpoint_path="checkpoint.pt",
            checkpoint_hash="hash",
            model_version="1.0",
            preprocessing_version="1.0",
        )


def test_model_prediction_rejects_invalid_confidence() -> None:
    with pytest.raises(
        ValueError,
        match="confidence must be between zero and one",
    ):
        ModelPrediction(
            predicted_class=0,
            predicted_label="Normal",
            probabilities=(0.8, 0.2),
            confidence=1.2,
            checkpoint_path="checkpoint.pt",
            checkpoint_hash="hash",
            model_version="1.0",
            preprocessing_version="1.0",
        )
