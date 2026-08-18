"""
notification_bridge.py — delivers governed decisions to the dashboard.

POSITION IN THE SYSTEM
----------------------
    Reasoning Pipeline   /api/v1/analyse      clinical analysis
            |
            v
    n8n + analyse_adapter                     governance, gating, HITL, audit
            |
            v
    THIS MODULE                               contract translation
            |
            v
    Supabase `agent_outputs`                  dashboard / notifications

Nothing reaches the dashboard except through governance. This module performs
the translation only: it makes no clinical judgement and no routing decision,
both of which have already been made upstream.

TWO DEFECTS FOUND IN THE EXISTING IMPORT PATH
---------------------------------------------
Both were verified against `scripts/agent_outputs.json` and
`scripts/import-agent-outputs.mjs` as they currently stand.

1. FIELD NESTING — the four clinical narrative fields import as NULL.
   `agent_outputs.json` nests them:

       { "agent_output": { "urgency_level": ..., "suggested_next_action": ...,
                           "doctor_technical_alert": ...,
                           "family_reassurance_message": ... } }

   but `import-agent-outputs.mjs` reads them at the root:

       urgency_level: item.urgency_level ?? null

   `item.urgency_level` is undefined, so all four columns import as null on
   every row. This module emits them at ROOT so the existing importer works
   unchanged, and also preserves the nested form for backward compatibility.

2. MISSPELLED KEY — the source data carries `dominullt_ecg_region`. The
   importer already defends against this with a fallback, but the misspelling
   should be corrected at source. This module emits the correct spelling.

WHAT THE CONTRACT IS MISSING
----------------------------
`agent_outputs` has no column for any governance decision — no
`human_review_required`, no `review_reasons`, no `gate_triggered`, no
`clearance_type`. As it stands the dashboard shows a prediction and an urgency
with no indication of whether the system cleared it, escalated it, or issued a
restricted clearance because the model cannot see the relevant class.

Until columns are added, everything governance-related is carried inside
`raw_output` (jsonb), which the importer already populates. See
REQUIRED_SCHEMA_ADDITIONS below for the columns to add.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Schema additions to request from the dashboard owner
# ---------------------------------------------------------------------------
REQUIRED_SCHEMA_ADDITIONS = {
    "human_review_required": "boolean - whether governance escalated this record",
    "clearance_type": "text - unqualified | restricted | none",
    "gate_triggered": "text - which governance gate fired, for counting escalations by cause",
    "governance_reason": "text - why, in clinician-readable form",
    "exclusion_statement": "text - what this model could NOT rule out; must be shown "
                           "beside any negative result",
    "audit_contract": "text - live-api-<sha> | offline-cohort; keeps evaluation "
                      "provenance from being conflated",
    "checkpoint_hash": "text - which model produced this, for traceability",
    "analysis_scope": "text - validated_mit_bih_compatible | exploratory_external_source "
                      "| unsupported",
}

# Governance decisions the dashboard must render differently.
CLEARANCE_UNQUALIFIED = "unqualified"
CLEARANCE_RESTRICTED = "restricted"
CLEARANCE_NONE = "none"


def _first_reason(governed: Dict[str, Any]) -> Optional[str]:
    reasons = governed.get("review_reasons") or []
    return reasons[0] if reasons else None


def _exclusion_statement(governed: Dict[str, Any]) -> Optional[str]:
    """The statement that must travel with any negative result.

    When the checkpoint cannot detect a class, a negative prediction is not an
    exclusion. The dashboard has to say so beside the result, or a clinician
    reading 'Normal Sinus Rhythm' will reasonably infer that abnormality has
    been ruled out.
    """
    if governed.get("clearance_type") != CLEARANCE_RESTRICTED:
        return None
    for reason in governed.get("review_reasons") or []:
        if "NOT been excluded" in reason or "RESTRICTED CLEARANCE" in reason:
            return reason
    return ("This model cannot exclude all arrhythmia classes. A negative "
            "result is not an exclusion.")


def to_agent_output_record(
    governed: Dict[str, Any],
    *,
    ehr_triage_note: Optional[str] = None,
    extracted_medical_entities: Optional[Dict[str, Any]] = None,
    true_label: Optional[str] = None,
    correct: Optional[bool] = None,
) -> Dict[str, Any]:
    """Translate one governed decision into the `agent_outputs` contract.

    `true_label` and `correct` are OPTIONAL and default to None. They are
    evaluation fields: the live API performs inference on raw ECG and returns
    no ground truth. They are populated only when replaying a labelled offline
    cohort. A dashboard row without them is the normal live case, not an error.
    """
    clearance = governed.get("clearance_type") or CLEARANCE_NONE
    reason = _first_reason(governed)
    exclusion = _exclusion_statement(governed)

    # Urgency comes from governance routing, NOT from a second clinical
    # opinion. The reasoning pipeline owns clinical interpretation; this maps
    # its decision onto the dashboard's existing vocabulary.
    if clearance == CLEARANCE_NONE and governed.get("human_review_required"):
        urgency = "High"
    elif clearance == CLEARANCE_RESTRICTED:
        urgency = "Review"
    else:
        urgency = "Low"

    record: Dict[str, Any] = {
        # ---- existing contract, emitted at ROOT so the current importer works
        "patient_id": governed.get("record_id"),
        "sample_id": None,
        "true_label": true_label,
        "prediction": governed.get("predicted_label"),
        "confidence": governed.get("prediction_confidence"),
        "correct": correct,
        "risk_flag": urgency,
        "dominant_ecg_region": None,   # live path uses Grad-CAM, not SHAP regions
        "ecg_lead": governed.get("lead_name"),
        "ehr_triage_note": ehr_triage_note,

        # ---- clinical content, VERBATIM from the reasoning pipeline
        "urgency_level": urgency,
        "suggested_next_action": governed.get("recommended_action"),
        "doctor_technical_alert": governed.get("doctor_report"),
        "family_reassurance_message": governed.get("next_of_kin_summary"),

        # ---- governance decision (currently lands in raw_output)
        "human_review_required": governed.get("human_review_required"),
        "clearance_type": clearance,
        "gate_triggered": governed.get("gate_triggered"),
        "governance_reason": reason,
        "exclusion_statement": exclusion,
        "audit_contract": governed.get("audit_contract"),
        "checkpoint_hash": governed.get("checkpoint_hash"),
        "analysis_scope": governed.get("analysis_scope"),

        "extracted_medical_entities": extracted_medical_entities or {},
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "success",

        # ---- backward compatibility with the nested shape
        "agent_output": {
            "urgency_level": urgency,
            "suggested_next_action": governed.get("recommended_action"),
            "doctor_technical_alert": governed.get("doctor_report"),
            "family_reassurance_message": governed.get("next_of_kin_summary"),
        },
    }
    return record


def to_agent_output_batch(
    governed_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return [to_agent_output_record(g) for g in governed_records]


def dashboard_display_rules() -> Dict[str, Any]:
    """How the dashboard must render each clearance type.

    This is a governance requirement, not a styling preference. A restricted
    clearance rendered identically to an unqualified one defeats the gate: the
    clinician sees a confident negative and reasonably infers exclusion.
    """
    return {
        CLEARANCE_UNQUALIFIED: {
            "meaning": "Governance cleared this record. No gate fired.",
            "display": "Normal routing. Show prediction and confidence.",
            "requires_exclusion_notice": False,
        },
        CLEARANCE_RESTRICTED: {
            "meaning": ("Cleared only for classes this checkpoint can detect. "
                        "One or more classes cannot be ruled out at any confidence."),
            "display": ("MUST show exclusion_statement beside the result. Must NOT "
                        "be rendered as a clean negative. A confident prediction "
                        "from a model with a blind class is not an exclusion."),
            "requires_exclusion_notice": True,
        },
        CLEARANCE_NONE: {
            "meaning": "Escalated to clinical review. A gate fired.",
            "display": "Show governance_reason and gate_triggered. Route to review queue.",
            "requires_exclusion_notice": False,
        },
    }
