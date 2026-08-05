from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from reasoning_pipeline.domain.enums.statuses import ConsistencyStatus
from reasoning_pipeline.domain.models.evidence_bundle import EvidenceBundle
from reasoning_pipeline.domain.models.evidence_item import EvidenceItem
from reasoning_pipeline.domain.models.reasoning_result import ReasoningResult
from reasoning_pipeline.reasoning.configuration import (
    ReasoningConfiguration,
)


@dataclass(frozen=True)
class ReasoningEngine:
    configuration: ReasoningConfiguration = ReasoningConfiguration()

    def reason(self, evidence: EvidenceBundle) -> ReasoningResult:
        supporting_weight = self._total_reliability(
            evidence.supporting_evidence
        )
        conflicting_weight = self._total_reliability(
            evidence.conflicting_evidence
        )

        total_directional_weight = (
            supporting_weight + conflicting_weight
        )

        evidence_balance = self._calculate_evidence_balance(
            supporting_weight=supporting_weight,
            conflicting_weight=conflicting_weight,
        )

        conflict_ratio = self._calculate_conflict_ratio(
            conflicting_weight=conflicting_weight,
            total_directional_weight=total_directional_weight,
        )

        average_reliability = self._average_reliability(
            (
                *evidence.supporting_evidence,
                *evidence.conflicting_evidence,
            )
        )

        consistency_status = self._determine_consistency_status(
            evidence=evidence,
            total_directional_weight=total_directional_weight,
            evidence_balance=evidence_balance,
            conflict_ratio=conflict_ratio,
            average_reliability=average_reliability,
        )

        reasoning_confidence = self._calculate_reasoning_confidence(
            evidence=evidence,
            consistency_status=consistency_status,
            average_reliability=average_reliability,
            conflict_ratio=conflict_ratio,
        )

        conclusion = self._build_conclusion(
            evidence=evidence,
            consistency_status=consistency_status,
        )

        limitations = self._build_limitations(
            evidence=evidence,
            consistency_status=consistency_status,
            average_reliability=average_reliability,
        )

        rule_trace = self._build_rule_trace(
            supporting_weight=supporting_weight,
            conflicting_weight=conflicting_weight,
            evidence_balance=evidence_balance,
            conflict_ratio=conflict_ratio,
            average_reliability=average_reliability,
            consistency_status=consistency_status,
        )

        return ReasoningResult(
            evidence=evidence,
            consistency_status=consistency_status,
            reasoning_confidence=reasoning_confidence,
            conclusion=conclusion,
            limitations=limitations,
            rule_trace=rule_trace,
            reasoning_version=self.configuration.reasoning_version,
        )

    def _determine_consistency_status(
        self,
        evidence: EvidenceBundle,
        total_directional_weight: float,
        evidence_balance: float,
        conflict_ratio: float,
        average_reliability: float,
    ) -> ConsistencyStatus:
        signal_quality = evidence.features.signal_quality.score

        if (
            signal_quality
            < self.configuration.low_signal_quality_threshold
        ):
            return ConsistencyStatus.LOW_SIGNAL_QUALITY

        if total_directional_weight == 0.0:
            return ConsistencyStatus.INSUFFICIENT_EVIDENCE

        if (
            average_reliability
            < self.configuration.low_reliability_threshold
        ):
            return ConsistencyStatus.INSUFFICIENT_EVIDENCE

        if (
            conflict_ratio
            >= self.configuration.conflict_ratio_threshold
        ):
            return ConsistencyStatus.CONFLICTING_EVIDENCE

        if (
            evidence_balance
            >= self.configuration.strong_support_threshold
            and len(evidence.supporting_evidence)
            >= self.configuration.minimum_strong_support_items
        ):
            return ConsistencyStatus.STRONGLY_SUPPORTED

        if (
            evidence_balance
            >= self.configuration.partial_support_threshold
        ):
            return ConsistencyStatus.PARTIALLY_SUPPORTED

        if evidence.conflicting_evidence:
            return ConsistencyStatus.CONFLICTING_EVIDENCE

        return ConsistencyStatus.INSUFFICIENT_EVIDENCE

    def _calculate_reasoning_confidence(
        self,
        evidence: EvidenceBundle,
        consistency_status: ConsistencyStatus,
        average_reliability: float,
        conflict_ratio: float,
    ) -> float:
        prediction_confidence = evidence.prediction.confidence

        evidence_factor = average_reliability

        conflict_factor = 1.0 - (
            conflict_ratio * self.configuration.conflict_penalty
        )

        status_factor = 1.0

        if (
            consistency_status
            == ConsistencyStatus.INSUFFICIENT_EVIDENCE
        ):
            status_factor = (
                self.configuration.insufficient_evidence_factor
            )

        elif (
            consistency_status
            == ConsistencyStatus.LOW_SIGNAL_QUALITY
        ):
            status_factor = (
                self.configuration.low_signal_quality_factor
            )

        confidence = (
            prediction_confidence
            * evidence_factor
            * conflict_factor
            * status_factor
        )

        return self._clamp(confidence)

    @staticmethod
    def _calculate_evidence_balance(
        supporting_weight: float,
        conflicting_weight: float,
    ) -> float:
        total_weight = supporting_weight + conflicting_weight

        if total_weight == 0.0:
            return 0.0

        return (
            supporting_weight - conflicting_weight
        ) / total_weight

    @staticmethod
    def _calculate_conflict_ratio(
        conflicting_weight: float,
        total_directional_weight: float,
    ) -> float:
        if total_directional_weight == 0.0:
            return 0.0

        return conflicting_weight / total_directional_weight

    @staticmethod
    def _total_reliability(
        items: Iterable[EvidenceItem],
    ) -> float:
        return sum(item.reliability for item in items)

    @staticmethod
    def _average_reliability(
        items: Iterable[EvidenceItem],
    ) -> float:
        items_tuple = tuple(items)

        if not items_tuple:
            return 0.0

        return sum(
            item.reliability for item in items_tuple
        ) / len(items_tuple)

    def _build_conclusion(
        self,
        evidence: EvidenceBundle,
        consistency_status: ConsistencyStatus,
    ) -> str:
        label = evidence.prediction.predicted_label

        conclusions = {
            ConsistencyStatus.STRONGLY_SUPPORTED: (
                f"The available evidence strongly supports the "
                f"federated model prediction of {label}."
            ),
            ConsistencyStatus.PARTIALLY_SUPPORTED: (
                f"The available evidence partially supports the "
                f"federated model prediction of {label}."
            ),
            ConsistencyStatus.CONFLICTING_EVIDENCE: (
                f"The available evidence contains meaningful "
                f"conflicts with the federated model prediction "
                f"of {label}."
            ),
            ConsistencyStatus.INSUFFICIENT_EVIDENCE: (
                f"There is insufficient reliable evidence to "
                f"confirm the federated model prediction of {label}."
            ),
            ConsistencyStatus.LOW_SIGNAL_QUALITY: (
                f"The federated model predicted {label}, but low "
                f"signal quality limits the reliability of this "
                f"conclusion."
            ),
            ConsistencyStatus.OUT_OF_SCOPE: (
                f"The prediction of {label} is outside the supported "
                f"scope of the current reasoning rules."
            ),
        }

        return conclusions[consistency_status]

    def _build_limitations(
        self,
        evidence: EvidenceBundle,
        consistency_status: ConsistencyStatus,
        average_reliability: float,
    ) -> tuple[str, ...]:
        limitations = list(evidence.limitations)

        if (
            consistency_status
            == ConsistencyStatus.CONFLICTING_EVIDENCE
        ):
            limitations.append(
                "Supporting and conflicting evidence were both present."
            )

        if (
            consistency_status
            == ConsistencyStatus.INSUFFICIENT_EVIDENCE
        ):
            limitations.append(
                "The available evidence was insufficient for a "
                "strong deterministic conclusion."
            )

        if (
            consistency_status
            == ConsistencyStatus.LOW_SIGNAL_QUALITY
        ):
            limitations.append(
                "Signal quality was below the configured reasoning "
                "threshold."
            )

        if (
            average_reliability
            < self.configuration.low_reliability_threshold
        ):
            limitations.append(
                "Average evidence reliability was below the "
                "configured threshold."
            )

        return self._unique_strings(limitations)

    def _build_rule_trace(
        self,
        supporting_weight: float,
        conflicting_weight: float,
        evidence_balance: float,
        conflict_ratio: float,
        average_reliability: float,
        consistency_status: ConsistencyStatus,
    ) -> tuple[str, ...]:
        return (
            (
                "Calculated supporting evidence weight as "
                f"{supporting_weight:.4f}."
            ),
            (
                "Calculated conflicting evidence weight as "
                f"{conflicting_weight:.4f}."
            ),
            (
                "Calculated normalised evidence balance as "
                f"{evidence_balance:.4f}."
            ),
            (
                "Calculated conflict ratio as "
                f"{conflict_ratio:.4f}."
            ),
            (
                "Calculated average evidence reliability as "
                f"{average_reliability:.4f}."
            ),
            (
                "Assigned consistency status "
                f"{consistency_status.value}."
            ),
        )

    @staticmethod
    def _unique_strings(
        values: Iterable[str],
    ) -> tuple[str, ...]:
        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            cleaned = value.strip()

            if not cleaned or cleaned in seen:
                continue

            seen.add(cleaned)
            result.append(cleaned)

        return tuple(result)

    @staticmethod
    def _clamp(value: float) -> float:
        return min(1.0, max(0.0, value))
