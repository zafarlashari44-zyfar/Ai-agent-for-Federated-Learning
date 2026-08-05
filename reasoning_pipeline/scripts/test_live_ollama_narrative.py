from __future__ import annotations

import json
from dataclasses import asdict

from reasoning_pipeline.domain.enums.statuses import ConsistencyStatus
from reasoning_pipeline.domain.models.clinical_report import ClinicalReport
from reasoning_pipeline.narrative import (
    NarrativeConfiguration,
    NarrativeGenerator,
)


def main() -> None:
    report = ClinicalReport(
        record_id="demo-record-001",
        predicted_label="Normal Sinus Rhythm",
        prediction_confidence=0.94,
        consistency_status=ConsistencyStatus.STRONGLY_SUPPORTED,
        reasoning_confidence=0.91,
        summary=(
            "The federated model predicted Normal Sinus Rhythm with "
            "94.0% confidence. The available evidence strongly supports "
            "the prediction."
        ),
        supporting_findings=(
            "The mean heart rate was within the configured normal range.",
            "The rhythm irregularity score was low.",
            "The mean QRS duration was within the expected range.",
        ),
        conflicting_findings=(),
        limitations=(
            "This output has not been independently reviewed by a clinician.",
        ),
        recommended_action=(
            "Review the prediction and supporting evidence through the "
            "standard clinical or research workflow."
        ),
        model_version="fl-model-v1",
        preprocessing_version="scribe-v2",
        evidence_version="evidence-builder-v1",
        reasoning_version="reasoning-engine-v1",
        report_version="clinical-report-v1",
        disclaimer=(
            "This automated report is intended for research and "
            "decision-support purposes only. It must not replace "
            "assessment by a qualified healthcare professional."
        ),
    )

    generator = NarrativeGenerator(
        configuration=NarrativeConfiguration(
            model_name="llama3.2:3b",
            host="http://localhost:11434",
            temperature=0.0,
            timeout_seconds=120.0,
            fallback_enabled=True,
        )
    )

    result = generator.generate(report)

    print(
        json.dumps(
            asdict(result),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
