from __future__ import annotations

import json
from typing import Any

from reasoning_pipeline.domain.models.clinical_report import (
    ClinicalReport,
)
from reasoning_pipeline.narrative.exceptions import (
    NarrativeValidationError,
)


class NarrativeResponseValidator:
    def parse_and_validate(
        self,
        *,
        raw_response: str,
        report: ClinicalReport,
    ) -> tuple[str, str]:
        doctor_report, next_of_kin_summary = self._parse_response(
            raw_response
        )

        self._validate_grounding(
            report=report,
            doctor_report=doctor_report,
            next_of_kin_summary=next_of_kin_summary,
        )

        return doctor_report, next_of_kin_summary

    @staticmethod
    def _parse_response(
        raw_response: str,
    ) -> tuple[str, str]:
        if not raw_response.strip():
            raise NarrativeValidationError(
                "Ollama returned an empty response"
            )

        parsed: Any = json.loads(raw_response)

        if not isinstance(parsed, dict):
            raise NarrativeValidationError(
                "Ollama response must be a JSON object"
            )

        allowed_keys = {
            "doctor_report",
            "next_of_kin_summary",
        }

        unexpected_keys = set(parsed) - allowed_keys

        if unexpected_keys:
            unexpected = ", ".join(
                sorted(str(key) for key in unexpected_keys)
            )
            raise NarrativeValidationError(
                "Ollama response contained unexpected fields: "
                f"{unexpected}"
            )

        doctor_report = parsed.get("doctor_report")
        next_of_kin_summary = parsed.get(
            "next_of_kin_summary"
        )

        if not isinstance(doctor_report, str):
            raise NarrativeValidationError(
                "doctor_report must be a string"
            )

        if not isinstance(next_of_kin_summary, str):
            raise NarrativeValidationError(
                "next_of_kin_summary must be a string"
            )

        doctor_report = doctor_report.strip()
        next_of_kin_summary = next_of_kin_summary.strip()

        if not doctor_report:
            raise NarrativeValidationError(
                "doctor_report cannot be empty"
            )

        if not next_of_kin_summary:
            raise NarrativeValidationError(
                "next_of_kin_summary cannot be empty"
            )

        return doctor_report, next_of_kin_summary

    @staticmethod
    def _validate_grounding(
        *,
        report: ClinicalReport,
        doctor_report: str,
        next_of_kin_summary: str,
    ) -> None:
        combined = (
            f"{doctor_report}\n{next_of_kin_summary}"
        ).casefold()

        if report.predicted_label.casefold() not in combined:
            raise NarrativeValidationError(
                "generated output omitted the predicted label"
            )

        recommendation_terms = {
            word.casefold().strip(".,:;()")
            for word in report.recommended_action.split()
            if len(word) >= 6
        }

        if recommendation_terms and not any(
            term in combined
            for term in recommendation_terms
        ):
            raise NarrativeValidationError(
                "generated output omitted the recommended action"
            )

        narrative_values = (
            doctor_report.strip(),
            next_of_kin_summary.strip(),
        )

        for narrative in narrative_values:
            if narrative.startswith("{") or narrative.startswith("["):
                raise NarrativeValidationError(
                    "generated narrative contained encoded JSON "
                    "instead of plain prose"
                )

        next_of_kin_text = next_of_kin_summary.casefold()

        prohibited_technical_terms = (
            "federated model",
            "federated learning",
            "scribe",
            "preprocessing",
            "evidence builder",
            "reasoning engine",
            "model version",
            "research workflow",
            "supporting evidence",
            "prediction confidence",
            "reasoning confidence",
        )

        if any(
            term in next_of_kin_text
            for term in prohibited_technical_terms
        ):
            raise NarrativeValidationError(
                "next-of-kin summary exposed internal technical details"
            )

        prohibited_recipient_instructions = (
            "review the prediction",
            "review the evidence",
            "review the supporting evidence",
            "clinical or research workflow",
        )

        if any(
            phrase in next_of_kin_text
            for phrase in prohibited_recipient_instructions
        ):
            raise NarrativeValidationError(
                "next-of-kin summary assigned an inappropriate "
                "technical action to the recipient"
            )

        prohibited_diagnostic_claims = (
            "confirmed diagnosis",
            "definitive diagnosis",
            "diagnosis is confirmed",
            "guaranteed diagnosis",
            "patient has been diagnosed",
            "the patient has a confirmed",
            "suffering from",
        )

        if any(
            claim in combined
            for claim in prohibited_diagnostic_claims
        ):
            raise NarrativeValidationError(
                "generated output presented the prediction as a "
                "confirmed diagnosis"
            )

        prohibited_recipient_wording = (
            "your ecg",
            "your heart",
            "you have a",
            "you have an",
            "you are diagnosed",
            "your diagnosis",
            "your condition",
            "your result",
        )

        if any(
            phrase in next_of_kin_text
            for phrase in prohibited_recipient_wording
        ):
            raise NarrativeValidationError(
                "next-of-kin summary addressed the recipient as "
                "though they were the patient"
            )
