from dataclasses import dataclass

import numpy as np

from reasoning_pipeline.domain.enums.statuses import OODStatus
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.domain.models.model_prediction import ModelPrediction
from reasoning_pipeline.domain.models.ood_assessment import OODAssessment
from reasoning_pipeline.domain.models.signal_suitability_assessment import (
    SignalSuitabilityAssessment,
)


@dataclass(frozen=True)
class OODThresholds:
    low_maximum_probability: float = 0.55
    high_normalized_entropy: float = 0.80
    high_q_class_proportion: float = 0.40
    low_beat_confidence: float = 0.60
    high_low_confidence_proportion: float = 0.50
    high_probability_instability: float = 0.20
    high_suitability_warning_count: int = 3
    large_resampling_ratio: float = 3.0
    uncertain_score: int = 2
    likely_ood_score: int = 4


class HeuristicOODAssessor:
    def __init__(self, thresholds: OODThresholds | None = None) -> None:
        self.thresholds = thresholds or OODThresholds()

    def assess(
        self,
        *,
        signal: ECGSignal,
        predictions: tuple[ModelPrediction, ...],
        suitability: SignalSuitabilityAssessment,
    ) -> OODAssessment:
        if not predictions:
            raise ValueError("OOD assessment requires beat predictions.")
        matrix = np.asarray([item.probabilities for item in predictions])
        maximum_probability = float(np.mean(np.max(matrix, axis=1)))
        entropy = -np.sum(matrix * np.log(np.clip(matrix, 1e-12, 1.0)), axis=1)
        normalized_entropy = float(np.mean(entropy) / np.log(matrix.shape[1]))
        q_proportion = float(
            np.mean([item.predicted_label == "Q" for item in predictions])
        )
        low_confidence = float(
            np.mean(
                [
                    item.confidence < self.thresholds.low_beat_confidence
                    for item in predictions
                ]
            )
        )
        instability = (
            float(np.mean(np.std(matrix, axis=0)))
            if len(predictions) > 1
            else 0.0
        )
        indicators: list[str] = []
        reasons: list[str] = []
        score = 0

        def add(condition: bool, points: int, indicator: str, reason: str) -> None:
            nonlocal score
            if condition:
                score += points
                indicators.append(indicator)
                reasons.append(reason)

        add(
            maximum_probability < self.thresholds.low_maximum_probability,
            2,
            "low_maximum_class_probability",
            "Mean maximum class probability is very low.",
        )
        add(
            normalized_entropy > self.thresholds.high_normalized_entropy,
            2,
            "high_prediction_entropy",
            "Prediction entropy is high.",
        )
        add(
            q_proportion > self.thresholds.high_q_class_proportion,
            2,
            "high_q_class_proportion",
            "A large proportion of beats were assigned to class Q.",
        )
        add(
            low_confidence > self.thresholds.high_low_confidence_proportion,
            1,
            "many_low_confidence_beats",
            "A large proportion of beat predictions have low confidence.",
        )
        add(
            instability > self.thresholds.high_probability_instability,
            1,
            "unstable_probabilities",
            "Class probabilities vary substantially across beats.",
        )
        add(
            len(suitability.warnings)
            >= self.thresholds.high_suitability_warning_count,
            1,
            "multiple_signal_quality_warnings",
            "The suitability assessment produced multiple warnings.",
        )
        external = not self._mit_bih_compatible(signal)
        add(
            external or signal.lead_name not in {"MLII", "II", "Lead II"},
            1,
            "unusual_source_or_lead",
            "Source or lead metadata differs from the validated model scope.",
        )
        source_rate = signal.original_sampling_rate_hz or signal.sampling_rate_hz
        ratio = max(source_rate, signal.sampling_rate_hz) / min(
            source_rate, signal.sampling_rate_hz
        )
        add(
            ratio >= self.thresholds.large_resampling_ratio,
            1,
            "large_resampling_ratio",
            "The source required a large sampling-rate conversion ratio.",
        )
        if score >= self.thresholds.likely_ood_score:
            status = OODStatus.LIKELY_OUT_OF_DISTRIBUTION
        elif score >= self.thresholds.uncertain_score:
            status = OODStatus.UNCERTAIN
        else:
            status = OODStatus.IN_DISTRIBUTION_LIKE
        warning = (
            ("Heuristic indicators suggest likely out-of-distribution input.",)
            if status is OODStatus.LIKELY_OUT_OF_DISTRIBUTION
            else ()
        )
        return OODAssessment(
            status=status,
            heuristic_score=score,
            maximum_class_probability=maximum_probability,
            normalized_prediction_entropy=normalized_entropy,
            q_class_proportion=q_proportion,
            low_confidence_beat_proportion=low_confidence,
            probability_instability=instability,
            indicators=tuple(indicators),
            reasons=tuple(reasons),
            warnings=warning,
        )

    @staticmethod
    def _mit_bih_compatible(signal: ECGSignal) -> bool:
        return "mit-bih" in signal.source.casefold() or (
            signal.source_format == "npy"
            and signal.lead_name in {None, "MLII", "II", "Lead II"}
        )
