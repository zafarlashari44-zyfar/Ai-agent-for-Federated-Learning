import pytest

from reasoning_pipeline.domain.enums.statuses import (
    OODStatus,
    SignalSuitabilityStatus,
)
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.domain.models.model_prediction import ModelPrediction
from reasoning_pipeline.domain.models.signal_suitability_assessment import (
    SignalSuitabilityAssessment,
)
from reasoning_pipeline.infrastructure.ood_assessment import HeuristicOODAssessor


def signal(source: str = "mit-bih") -> ECGSignal:
    return ECGSignal(
        record_id="record",
        samples=(0.0, 1.0, 0.0),
        sampling_rate_hz=360.0,
        source=source,
        lead_name="MLII",
        source_format="npy",
        units="mV",
    )


def suitability(warnings: tuple[str, ...] = ()) -> SignalSuitabilityAssessment:
    return SignalSuitabilityAssessment(
        status=(
            SignalSuitabilityStatus.ACCEPTED_WITH_WARNINGS
            if warnings
            else SignalSuitabilityStatus.ACCEPTED
        ),
        suitable_for_processing=True,
        quality_score=0.9,
        duration_seconds=10.0,
        sampling_rate_hz=360.0,
        selected_lead="MLII",
        units="mV",
        detected_r_peak_count=10,
        estimated_heart_rate_bpm=60.0,
        finite_sample_ratio=1.0,
        flatline_percentage=0.0,
        clipping_percentage=0.0,
        noise_score=0.1,
        warnings=warnings,
    )


def prediction(probabilities: tuple[float, ...]) -> ModelPrediction:
    maximum = max(probabilities)
    predicted = probabilities.index(maximum)
    labels = ("N", "S", "V", "F", "Q")
    return ModelPrediction(
        predicted_class=predicted,
        predicted_label=labels[predicted],
        probabilities=probabilities,
        confidence=maximum,
        checkpoint_path="model.pth",
        checkpoint_hash="hash",
        model_version="model",
        preprocessing_version="preprocessing",
    )


def test_confident_mit_bih_predictions_are_distribution_like() -> None:
    result = HeuristicOODAssessor().assess(
        signal=signal(),
        predictions=(prediction((0.9, 0.025, 0.025, 0.025, 0.025)),) * 4,
        suitability=suitability(),
    )
    assert result.status is OODStatus.IN_DISTRIBUTION_LIKE


def test_low_confidence_high_entropy_predictions_are_likely_ood() -> None:
    result = HeuristicOODAssessor().assess(
        signal=signal("private-ecg"),
        predictions=(prediction((0.22, 0.21, 0.20, 0.19, 0.18)),) * 5,
        suitability=suitability(("one", "two", "three")),
    )
    assert result.maximum_class_probability < 0.55
    assert result.normalized_prediction_entropy > 0.8
    assert result.status is OODStatus.LIKELY_OUT_OF_DISTRIBUTION


def test_high_q_class_proportion_is_reported() -> None:
    q = prediction((0.05, 0.05, 0.05, 0.05, 0.8))
    normal = prediction((0.8, 0.05, 0.05, 0.05, 0.05))
    result = HeuristicOODAssessor().assess(
        signal=signal(),
        predictions=(q, q, q, normal),
        suitability=suitability(),
    )
    assert result.q_class_proportion == pytest.approx(0.75)
    assert "high_q_class_proportion" in result.indicators
