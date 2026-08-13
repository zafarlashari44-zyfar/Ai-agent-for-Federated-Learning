from __future__ import annotations

import json
from collections.abc import Iterable

from ollama import ResponseError

from reasoning_pipeline.domain.models.clinical_report import (
    ClinicalReport,
)
from reasoning_pipeline.domain.models.narrative_result import (
    NarrativeResult,
)
from reasoning_pipeline.narrative.configuration import (
    NarrativeConfiguration,
)
from reasoning_pipeline.narrative.exceptions import (
    NarrativeGenerationError,
    NarrativeValidationError,
)
from reasoning_pipeline.narrative.ollama_client import (
    NarrativeModelClient,
    OllamaClientAdapter,
)
from reasoning_pipeline.narrative.prompt_builder import (
    NarrativePromptBuilder,
)
from reasoning_pipeline.narrative.response_validator import (
    NarrativeResponseValidator,
)


class NarrativeGenerator:
    def __init__(
        self,
        configuration: NarrativeConfiguration | None = None,
        client: NarrativeModelClient | None = None,
        prompt_builder: NarrativePromptBuilder | None = None,
        response_validator: NarrativeResponseValidator | None = None,
    ) -> None:
        self._configuration = (
            configuration or NarrativeConfiguration()
        )
        self._client = client or OllamaClientAdapter(
            self._configuration
        )
        self._prompt_builder = (
            prompt_builder or NarrativePromptBuilder()
        )
        self._response_validator = (
            response_validator or NarrativeResponseValidator()
        )

    def generate(
        self,
        report: ClinicalReport,
    ) -> NarrativeResult:
        messages = self._prompt_builder.build_messages(report)

        try:
            raw_response = self._client.generate(messages)

            doctor_report, next_of_kin_summary = (
                self._response_validator.parse_and_validate(
                    raw_response=raw_response,
                    report=report,
                )
            )

            return NarrativeResult(
                record_id=report.record_id,
                doctor_report=doctor_report,
                next_of_kin_summary=next_of_kin_summary,
                provider="ollama",
                model_name=self._configuration.model_name,
                prompt_version=self._configuration.prompt_version,
                fallback_used=False,
            )

        except (
            ConnectionError,
            OSError,
            RuntimeError,
            ResponseError,
            json.JSONDecodeError,
            NarrativeValidationError,
            TypeError,
        ) as error:
            if not self._configuration.fallback_enabled:
                raise NarrativeGenerationError(
                    "Ollama narrative generation failed"
                ) from error

            return self._build_fallback(
                report=report,
                warning=(
                    "Ollama generation failed; deterministic fallback "
                    f"was used: {type(error).__name__}"
                ),
            )

    def _build_fallback(
        self,
        *,
        report: ClinicalReport,
        warning: str,
    ) -> NarrativeResult:
        supporting = self._format_findings(
            report.supporting_findings,
            default="No supporting findings were available.",
        )
        conflicting = self._format_findings(
            report.conflicting_findings,
            default="No conflicting findings were identified.",
        )
        limitations = self._format_findings(
            report.limitations,
            default="No additional limitations were recorded.",
        )

        doctor_report = (
            f"Automated ECG model prediction: "
            f"{report.predicted_label}. "
            f"Prediction confidence: "
            f"{report.prediction_confidence:.1%}. "
            f"Deterministic reasoning status: "
            f"{report.consistency_status.value.replace('_', ' ')}. "
            f"Reasoning confidence: "
            f"{report.reasoning_confidence:.1%}. "
            f"Supporting findings: {supporting} "
            f"Conflicting findings: {conflicting} "
            f"Limitations: {limitations} "
            f"Recommended action: {report.recommended_action} "
            f"{report.disclaimer}"
        )

        next_of_kin_summary = (
            "An automated assessment of the patient's ECG identified "
            f"a pattern consistent with {report.predicted_label}. "
            "This is not a confirmed diagnosis. "
            "The clinical team will review the result and determine "
            "whether any further assessment is needed. "
            "No immediate action is required from the family unless "
            "the healthcare team provides further instructions."
        )


        return NarrativeResult(
            record_id=report.record_id,
            doctor_report=doctor_report,
            next_of_kin_summary=next_of_kin_summary,
            provider="deterministic-fallback",
            model_name=self._configuration.model_name,
            prompt_version=self._configuration.prompt_version,
            fallback_used=True,
            warnings=(warning,),
        )

    @staticmethod
    def _format_findings(
        findings: Iterable[str],
        *,
        default: str,
    ) -> str:
        cleaned = tuple(
            finding.strip()
            for finding in findings
            if finding.strip()
        )

        if not cleaned:
            return default

        return "; ".join(cleaned) + "."
