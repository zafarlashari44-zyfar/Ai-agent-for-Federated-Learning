from __future__ import annotations

import json

from reasoning_pipeline.domain.models.clinical_report import ClinicalReport


class NarrativePromptBuilder:
    SYSTEM_PROMPT = """
You are a medical communication assistant inside an ECG
decision-support pipeline.

The structured clinical report is the only source of truth.

You must produce exactly two plain-text narratives:
1. a doctor-facing clinical narrative
2. a next-of-kin-facing summary

General rules:
- Do not invent diagnoses, symptoms, measurements, treatments, or risks.
- Do not change the predicted ECG label.
- Do not present the result as a confirmed diagnosis.
- Do not output nested JSON, markdown, headings, lists, or code blocks.
- Each field must contain normal readable prose only.
- Do not repeat internal system architecture or implementation details.

Doctor narrative rules:
- State that the result is an automated model prediction.
- Include the predicted label.
- Include prediction confidence and reasoning confidence.
- Summarise the supporting and conflicting clinical findings.
- Include relevant limitations.
- Include the recommended clinical action.
- Use concise, professional, clinically neutral language.
- Do not include model version names unless explicitly required.

Next-of-kin summary rules:
- Write for a relative or authorised representative of the patient.
- Refer to the individual as "the patient".
- Do not address the recipient as the patient.
- Use calm, respectful, plain language.
- Explain that an automated ECG assessment was performed.
- State that the result is not a confirmed diagnosis.
- Explain what the clinical team will do next.
- State clearly whether any action is required from the family.
- Do not mention federated learning, Scribe, preprocessing, evidence,
  reasoning engines, model versions, confidence percentages, JSON,
  research workflow, or technical review processes.
- Do not ask the next of kin to review predictions or evidence.
- Do not say "your ECG", "you have", or "your diagnosis".

Return exactly one JSON object with these two string fields:
- "doctor_report"
- "next_of_kin_summary"

The values must be plain prose strings, not JSON encoded inside strings.
""".strip()

    def build_messages(
        self,
        report: ClinicalReport,
    ) -> list[dict[str, str]]:
        payload = {
            "record_id": report.record_id,
            "predicted_label": report.predicted_label,
            "prediction_confidence": report.prediction_confidence,
            "consistency_status": report.consistency_status.value,
            "reasoning_confidence": report.reasoning_confidence,
            "summary": report.summary,
            "supporting_findings": list(report.supporting_findings),
            "conflicting_findings": list(report.conflicting_findings),
            "limitations": list(report.limitations),
            "recommended_action": report.recommended_action,
            "disclaimer": report.disclaimer,
        }

        user_prompt = (
            "Create the doctor narrative and next-of-kin summary from "
            "the following deterministic ECG report.\n\n"
            "For the next-of-kin summary, translate the clinical result "
            "into plain language. Do not mention system architecture, "
            "model type, preprocessing, research workflow, confidence "
            "percentages, or supporting evidence review. Explain what "
            "the clinical team should do next and whether the family "
            "needs to take any action.\n\n"
            f"Clinical report:\n{json.dumps(payload, indent=2)}"
        )

        return [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]
