from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from reasoning_pipeline.domain.enums.statuses import EvidenceDirection
from reasoning_pipeline.domain.models.evidence_bundle import EvidenceBundle
from reasoning_pipeline.domain.models.evidence_item import EvidenceItem
from reasoning_pipeline.domain.models.feature_set import FeatureSet
from reasoning_pipeline.domain.models.model_prediction import (
    ModelPrediction,
)
from reasoning_pipeline.evidence.configuration import (
    EvidenceBuilderConfiguration,
)


@dataclass(frozen=True)
class EvidenceBuilder:
    configuration: EvidenceBuilderConfiguration = (
        EvidenceBuilderConfiguration()
    )

    def build(
        self,
        record_id: str,
        prediction: ModelPrediction,
        features: FeatureSet,
    ) -> EvidenceBundle:
        if not record_id.strip():
            raise ValueError("record_id cannot be empty")

        items: list[EvidenceItem] = []
        limitations: list[str] = []

        predicted_family = self._normalise_prediction_label(
            prediction.predicted_label
        )

        items.append(
            self._build_model_prediction_evidence(prediction)
        )

        items.append(
            self._build_signal_quality_evidence(
                features=features,
                predicted_family=predicted_family,
            )
        )

        self._append_heart_rate_evidence(
            items=items,
            limitations=limitations,
            features=features,
            predicted_family=predicted_family,
        )

        self._append_irregularity_evidence(
            items=items,
            limitations=limitations,
            features=features,
            predicted_family=predicted_family,
        )

        self._append_qrs_evidence(
            items=items,
            limitations=limitations,
            features=features,
            predicted_family=predicted_family,
        )

        self._append_abnormal_beat_evidence(
            items=items,
            limitations=limitations,
            features=features,
            predicted_family=predicted_family,
        )

        self._append_pipeline_limitations(
            limitations=limitations,
            prediction=prediction,
            features=features,
        )

        supporting = tuple(
            item
            for item in items
            if item.direction == EvidenceDirection.SUPPORTS
        )
        conflicting = tuple(
            item
            for item in items
            if item.direction == EvidenceDirection.CONTRADICTS
        )
        neutral = tuple(
            item
            for item in items
            if item.direction
            not in {
                EvidenceDirection.SUPPORTS,
                EvidenceDirection.CONTRADICTS,
            }
        )

        return EvidenceBundle(
            record_id=record_id,
            prediction=prediction,
            features=features,
            supporting_evidence=supporting,
            conflicting_evidence=conflicting,
            neutral_evidence=neutral,
            limitations=self._unique_strings(limitations),
            evidence_version=self.configuration.evidence_version,
        )

    def _build_model_prediction_evidence(
        self,
        prediction: ModelPrediction,
    ) -> EvidenceItem:
        return EvidenceItem(
            evidence_id="model.prediction",
            feature_name="model_prediction",
            measured_value=prediction.predicted_label,
            unit=None,
            interpretation=(
                f"The federated model predicted "
                f"{prediction.predicted_label} with "
                f"{prediction.confidence:.1%} confidence."
            ),
            direction=EvidenceDirection.SUPPORTS,
            reliability=prediction.confidence,
            source_reference="model_prediction.predicted_label",
        )

    def _build_signal_quality_evidence(
        self,
        features: FeatureSet,
        predicted_family: str,
    ) -> EvidenceItem:
        quality = features.signal_quality
        direction = EvidenceDirection.NEUTRAL

        if predicted_family == "unknown":
            if (
                quality.score
                < self.configuration.low_signal_quality_threshold
            ):
                direction = EvidenceDirection.SUPPORTS

        return EvidenceItem(
            evidence_id="signal.quality",
            feature_name="signal_quality",
            measured_value=quality.score,
            unit=None,
            interpretation=(
                f"Signal quality was classified as "
                f"{quality.status.value} with a score of "
                f"{quality.score:.2f}."
            ),
            direction=direction,
            reliability=quality.score,
            source_reference="features.signal_quality.score",
        )

    def _append_heart_rate_evidence(
        self,
        items: list[EvidenceItem],
        limitations: list[str],
        features: FeatureSet,
        predicted_family: str,
    ) -> None:
        heart_rate = features.rhythm.heart_rate_mean_bpm

        if heart_rate is None:
            limitations.append(
                "Mean heart rate could not be calculated."
            )
            items.append(
                self._unavailable_item(
                    evidence_id="rhythm.heart_rate",
                    feature_name="heart_rate_mean_bpm",
                    interpretation=(
                        "Mean heart rate was unavailable."
                    ),
                    source_reference=(
                        "features.rhythm.heart_rate_mean_bpm"
                    ),
                )
            )
            return

        lower = self.configuration.normal_heart_rate_min_bpm
        upper = self.configuration.normal_heart_rate_max_bpm
        is_normal_rate = lower <= heart_rate <= upper

        direction = EvidenceDirection.NEUTRAL
        interpretation = (
            f"Mean heart rate was {heart_rate:.1f} beats per minute."
        )

        if predicted_family == "normal":
            direction = (
                EvidenceDirection.SUPPORTS
                if is_normal_rate
                else EvidenceDirection.CONTRADICTS
            )
        elif predicted_family == "supraventricular":
            if heart_rate > upper:
                direction = EvidenceDirection.SUPPORTS
        elif predicted_family == "unknown":
            direction = EvidenceDirection.NEUTRAL

        reliability = self._rhythm_reliability(features)

        items.append(
            EvidenceItem(
                evidence_id="rhythm.heart_rate",
                feature_name="heart_rate_mean_bpm",
                measured_value=heart_rate,
                unit="bpm",
                interpretation=interpretation,
                direction=direction,
                reliability=reliability,
                source_reference=(
                    "features.rhythm.heart_rate_mean_bpm"
                ),
            )
        )

    def _append_irregularity_evidence(
        self,
        items: list[EvidenceItem],
        limitations: list[str],
        features: FeatureSet,
        predicted_family: str,
    ) -> None:
        score = features.rhythm.irregularity_score

        if score is None:
            limitations.append(
                "Rhythm irregularity could not be assessed."
            )
            items.append(
                self._unavailable_item(
                    evidence_id="rhythm.irregularity",
                    feature_name="irregularity_score",
                    interpretation=(
                        "Rhythm irregularity score was unavailable."
                    ),
                    source_reference=(
                        "features.rhythm.irregularity_score"
                    ),
                )
            )
            return

        is_irregular = (
            score >= self.configuration.irregularity_threshold
        )

        direction = EvidenceDirection.NEUTRAL

        if predicted_family == "normal":
            direction = (
                EvidenceDirection.CONTRADICTS
                if is_irregular
                else EvidenceDirection.SUPPORTS
            )
        elif predicted_family in {
            "supraventricular",
            "ventricular",
        }:
            if is_irregular:
                direction = EvidenceDirection.SUPPORTS

        interpretation = (
            f"Rhythm irregularity score was {score:.2f}; "
            f"the configured threshold was "
            f"{self.configuration.irregularity_threshold:.2f}."
        )

        items.append(
            EvidenceItem(
                evidence_id="rhythm.irregularity",
                feature_name="irregularity_score",
                measured_value=score,
                unit=None,
                interpretation=interpretation,
                direction=direction,
                reliability=self._rhythm_reliability(features),
                source_reference=(
                    "features.rhythm.irregularity_score"
                ),
            )
        )

    def _append_qrs_evidence(
        self,
        items: list[EvidenceItem],
        limitations: list[str],
        features: FeatureSet,
        predicted_family: str,
    ) -> None:
        qrs_duration = features.morphology.mean_qrs_duration_ms

        if qrs_duration is None:
            limitations.append(
                "Mean QRS duration could not be estimated."
            )
            items.append(
                self._unavailable_item(
                    evidence_id="morphology.qrs_duration",
                    feature_name="mean_qrs_duration_ms",
                    interpretation=(
                        "Mean QRS duration was unavailable."
                    ),
                    source_reference=(
                        "features.morphology.mean_qrs_duration_ms"
                    ),
                )
            )
            return

        wide_qrs = (
            qrs_duration
            >= self.configuration.wide_qrs_threshold_ms
        )

        direction = EvidenceDirection.NEUTRAL

        if predicted_family == "normal":
            direction = (
                EvidenceDirection.CONTRADICTS
                if wide_qrs
                else EvidenceDirection.SUPPORTS
            )
        elif predicted_family == "ventricular":
            direction = (
                EvidenceDirection.SUPPORTS
                if wide_qrs
                else EvidenceDirection.CONTRADICTS
            )

        interpretation = (
            f"Mean QRS duration was {qrs_duration:.1f} ms; "
            f"the configured wide-QRS threshold was "
            f"{self.configuration.wide_qrs_threshold_ms:.1f} ms."
        )

        items.append(
            EvidenceItem(
                evidence_id="morphology.qrs_duration",
                feature_name="mean_qrs_duration_ms",
                measured_value=qrs_duration,
                unit="ms",
                interpretation=interpretation,
                direction=direction,
                reliability=self._morphology_reliability(features),
                source_reference=(
                    "features.morphology.mean_qrs_duration_ms"
                ),
            )
        )

    def _append_abnormal_beat_evidence(
        self,
        items: list[EvidenceItem],
        limitations: list[str],
        features: FeatureSet,
        predicted_family: str,
    ) -> None:
        abnormal_count = features.morphology.abnormal_beat_count

        if abnormal_count is None:
            limitations.append(
                "Abnormal beat count could not be estimated."
            )
            items.append(
                self._unavailable_item(
                    evidence_id="morphology.abnormal_beats",
                    feature_name="abnormal_beat_count",
                    interpretation=(
                        "Abnormal beat count was unavailable."
                    ),
                    source_reference=(
                        "features.morphology.abnormal_beat_count"
                    ),
                )
            )
            return

        direction = EvidenceDirection.NEUTRAL

        if predicted_family == "normal":
            direction = (
                EvidenceDirection.SUPPORTS
                if abnormal_count == 0
                else EvidenceDirection.CONTRADICTS
            )
        elif predicted_family in {"ventricular", "fusion"}:
            if abnormal_count > 0:
                direction = EvidenceDirection.SUPPORTS

        items.append(
            EvidenceItem(
                evidence_id="morphology.abnormal_beats",
                feature_name="abnormal_beat_count",
                measured_value=abnormal_count,
                unit="beats",
                interpretation=(
                    f"The morphology extractor identified "
                    f"{abnormal_count} abnormal beat(s)."
                ),
                direction=direction,
                reliability=self._morphology_reliability(features),
                source_reference=(
                    "features.morphology.abnormal_beat_count"
                ),
            )
        )

    def _append_pipeline_limitations(
        self,
        limitations: list[str],
        prediction: ModelPrediction,
        features: FeatureSet,
    ) -> None:
        if (
            prediction.confidence
            < self.configuration.low_model_confidence_threshold
        ):
            limitations.append(
                "Model confidence was below the configured "
                "confidence threshold."
            )

        if (
            features.signal_quality.score
            < self.configuration.low_signal_quality_threshold
        ):
            limitations.append(
                "Low signal quality may reduce the reliability "
                "of extracted evidence."
            )

        if features.morphology.mean_pr_interval_ms is None:
            limitations.append(
                "PR interval evidence was unavailable."
            )

        if features.morphology.mean_qt_interval_ms is None:
            limitations.append(
                "QT interval evidence was unavailable."
            )

        limitations.extend(features.warnings)
        limitations.extend(features.signal_quality.warnings)

    def _rhythm_reliability(
        self,
        features: FeatureSet,
    ) -> float:
        return self._mean_reliability(
            (
                features.signal_quality.score,
                features.r_peaks.confidence,
            )
        )

    def _morphology_reliability(
        self,
        features: FeatureSet,
    ) -> float:
        return self._mean_reliability(
            (
                features.signal_quality.score,
                features.r_peaks.confidence,
                features.morphology.morphology_confidence,
            )
        )

    @staticmethod
    def _mean_reliability(values: Iterable[float]) -> float:
        values_tuple = tuple(values)

        if not values_tuple:
            return 0.0

        result = sum(values_tuple) / len(values_tuple)
        return min(1.0, max(0.0, result))

    @staticmethod
    def _unavailable_item(
        evidence_id: str,
        feature_name: str,
        interpretation: str,
        source_reference: str,
    ) -> EvidenceItem:
        return EvidenceItem(
            evidence_id=evidence_id,
            feature_name=feature_name,
            measured_value=None,
            unit=None,
            interpretation=interpretation,
            direction=EvidenceDirection.UNAVAILABLE,
            reliability=0.0,
            source_reference=source_reference,
        )

    @staticmethod
    def _normalise_prediction_label(label: str) -> str:
        normalised = (
            label.strip()
            .lower()
            .replace("-", " ")
            .replace("_", " ")
        )

        aliases = {
            "n": "normal",
            "normal": "normal",
            "normal sinus rhythm": "normal",
            "normal beat": "normal",
            "s": "supraventricular",
            "supraventricular": "supraventricular",
            "supraventricular ectopic": "supraventricular",
            "supraventricular ectopic beat": "supraventricular",
            "v": "ventricular",
            "ventricular": "ventricular",
            "ventricular arrhythmia": "ventricular",
            "ventricular ectopic beat": "ventricular",
            "f": "fusion",
            "fusion": "fusion",
            "fusion beat": "fusion",
            "q": "unknown",
            "unknown": "unknown",
            "unclassifiable": "unknown",
            "unknown unclassifiable": "unknown",
        }

        return aliases.get(normalised, "unknown")

    @staticmethod
    def _unique_strings(values: Iterable[str]) -> tuple[str, ...]:
        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            cleaned = value.strip()

            if not cleaned or cleaned in seen:
                continue

            seen.add(cleaned)
            result.append(cleaned)

        return tuple(result)
