from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from reasoning_pipeline.domain.enums.statuses import ConsistencyStatus
from reasoning_pipeline.domain.models.clinical_report import ClinicalReport
from reasoning_pipeline.domain.models.evidence_item import EvidenceItem
from reasoning_pipeline.domain.models.reasoning_result import ReasoningResult
from reasoning_pipeline.reporting.configuration import ReportConfiguration


@dataclass(frozen=True)
class ReportGenerator:
    configuration: ReportConfiguration = ReportConfiguration()

    def generate(
        self,
        reasoning_result: ReasoningResult,
    ) -> ClinicalReport:
        evidence = reasoning_result.evidence
        prediction = evidence.prediction

        supporting_findings = self._extract_findings(
            evidence.supporting_evidence
        )
        conflicting_findings = self._extract_findings(
            evidence.conflicting_evidence
        )

        return ClinicalReport(
            record_id=evidence.record_id,
            predicted_label=prediction.predicted_label,
            prediction_confidence=prediction.confidence,
            consistency_status=reasoning_result.consistency_status,
            reasoning_confidence=(
                reasoning_result.reasoning_confidence
            ),
            summary=self._build_summary(reasoning_result),
            supporting_findings=supporting_findings,
            conflicting_findings=conflicting_findings,
            limitations=self._unique_strings(
                reasoning_result.limitations
            ),
            recommended_action=self._recommended_action(
                reasoning_result.consistency_status
            ),
            model_version=prediction.model_version,
            preprocessing_version=prediction.preprocessing_version,
            evidence_version=evidence.evidence_version,
            reasoning_version=reasoning_result.reasoning_version,
            report_version=self.configuration.report_version,
            disclaimer=self.configuration.disclaimer,
        )

    @staticmethod
    def _build_summary(
        reasoning_result: ReasoningResult,
    ) -> str:
        prediction = reasoning_result.evidence.prediction

        return (
            f"The federated model predicted "
            f"{prediction.predicted_label} with "
            f"{prediction.confidence:.1%} confidence. "
            f"{reasoning_result.conclusion} "
            f"The deterministic reasoning confidence was "
            f"{reasoning_result.reasoning_confidence:.1%}."
        )

    @staticmethod
    def _recommended_action(
        status: ConsistencyStatus,
    ) -> str:
        actions = {
            ConsistencyStatus.STRONGLY_SUPPORTED: (
                "Review the prediction and supporting evidence through "
                "the standard clinical or research workflow."
            ),
            ConsistencyStatus.PARTIALLY_SUPPORTED: (
                "Review the prediction alongside the available evidence "
                "before drawing a conclusion."
            ),
            ConsistencyStatus.CONFLICTING_EVIDENCE: (
                "Manual review is recommended because the extracted "
                "evidence does not consistently agree with the prediction."
            ),
            ConsistencyStatus.INSUFFICIENT_EVIDENCE: (
                "Obtain or review additional reliable ECG evidence before "
                "using the prediction."
            ),
            ConsistencyStatus.LOW_SIGNAL_QUALITY: (
                "Review signal acquisition and consider repeating or "
                "cleaning the ECG before relying on the prediction."
            ),
            ConsistencyStatus.OUT_OF_SCOPE: (
                "Refer the record for manual review because it is outside "
                "the supported scope of the current reasoning rules."
            ),
        }

        return actions[status]

    @staticmethod
    def _extract_findings(
        items: Iterable[EvidenceItem],
    ) -> tuple[str, ...]:
        return tuple(
            item.interpretation.strip()
            for item in items
            if item.interpretation.strip()
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
