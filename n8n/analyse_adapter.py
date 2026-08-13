"""
analyse_adapter.py — maps the ECG Reasoning Pipeline's /api/v1/analyse
response onto the fields the n8n governance layer needs.

RESPONSIBILITY BOUNDARY
-----------------------
The Reasoning Pipeline is the single source of clinical truth. This module
performs NO clinical interpretation, NO reasoning, NO narrative generation and
NO explainability. It reads the API response and derives only the operational
signals governance needs to make routing decisions.

Every field below is one of three kinds, and each is labelled in
`field_provenance` on the output so the audit record can never blur them:

  api        - taken verbatim from the API response
  derived    - computed from API values by a stated formula in this file
  governance - a policy decision belonging to this layer, not clinical fact

WHAT THIS REPLACED
------------------
The previous FastAPI bridge (n8n/app.py) called a local LLM to produce urgency
levels, clinical status and next actions. All of that now comes from the API:
`reasoning`, `clinical_report` and `narrative`. Nothing in this file generates
clinical content.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# GOVERNANCE POLICY
#
# These are the only numbers this layer owns. They are policy, not clinical
# fact, and they are stated here rather than buried in a workflow node so a
# reviewer can find and challenge them.
# ---------------------------------------------------------------------------
POLICY_VERSION = "governance-3.0-api-integrated"

# Classes the frozen federated checkpoint cannot emit (recall 0.0 in the
# team's own per-class evaluation). Verified against the deployed model by
# checking emitted probabilities - see verify_blind_classes().
DECLARED_BLIND_LABELS = ("S", "F")

CONFIDENCE_FLOOR = 0.70          # below this, a prediction is not acted on alone
ENTROPY_CEILING = 0.55           # normalised; above this the model is unsure
ABNORMAL_BURDEN_CEILING = 0.20   # fraction of beats abnormal before escalation
RELIABILITY_FLOOR = 0.50         # evidence items below this are discounted


# ---------------------------------------------------------------------------
# Derivations
# ---------------------------------------------------------------------------
def normalised_entropy(probabilities: Sequence[float]) -> Optional[float]:
    """Shannon entropy over the class distribution, scaled to 0..1.

    DERIVED. Replaces the MC-Dropout uncertainty band the previous pipeline
    read from a CSV artefact. This is strictly better: it is computed from the
    probabilities the deployed model actually emitted for this recording,
    rather than from a separate offline run that may not correspond to the
    checkpoint in use.
    """
    probs = [p for p in probabilities if p is not None]
    if len(probs) < 2:
        return None
    total = sum(probs)
    if total <= 0:
        return None
    probs = [p / total for p in probs]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return round(entropy / math.log(len(probs)), 4)


def verify_blind_classes(
    probabilities: Sequence[float],
    class_labels: Sequence[str],
) -> Dict[str, Any]:
    """Check the declared blind classes against what the model actually emits.

    The previous governance layer ASSUMED S and F were unreachable, on the
    basis of an evaluation CSV. This verifies the assumption against the
    deployed checkpoint: if a declared-blind class carries a non-trivial
    probability mass, the assumption is stale and the audit record says so
    rather than silently applying a wrong gate.
    """
    if not probabilities or not class_labels:
        return {"verified": False, "reason": "probabilities or labels absent"}

    mass: Dict[str, float] = {}
    for label, prob in zip(class_labels, probabilities):
        if label in DECLARED_BLIND_LABELS:
            mass[label] = round(float(prob), 6)

    if not mass:
        return {
            "verified": False,
            "reason": "declared blind labels not present in this label set",
            "declared": list(DECLARED_BLIND_LABELS),
            "labels_seen": list(class_labels),
        }

    still_blind = all(v < 0.01 for v in mass.values())
    return {
        "verified": True,
        "declared_blind_labels": list(DECLARED_BLIND_LABELS),
        "probability_mass": mass,
        "assumption_holds": still_blind,
        "note": (
            "Blind-class assumption confirmed against deployed model output."
            if still_blind
            else "WARNING: a declared-blind class carries probability mass. "
                 "The recall-0.0 assumption may be stale for this checkpoint."
        ),
    }


def _beat_entropies(beat_results: Sequence[Dict[str, Any]]) -> List[float]:
    out = []
    for beat in beat_results:
        pred = beat.get("prediction") or {}
        ent = normalised_entropy(pred.get("probabilities") or [])
        if ent is not None:
            out.append(ent)
    return out


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------
def adapt(
    response: Dict[str, Any],
    *,
    class_labels: Sequence[str] = ("N", "S", "V", "F", "Q"),
) -> Dict[str, Any]:
    """Map one /api/v1/analyse response onto governance inputs.

    Returns a flat dict the n8n governance node consumes, plus a
    `field_provenance` map recording where every value came from.
    """
    signal = response.get("signal") or {}
    prediction = response.get("prediction") or {}
    summary = response.get("recording_summary") or {}
    reasoning = response.get("reasoning") or {}
    report = response.get("clinical_report") or {}
    narrative = response.get("narrative") or {}
    evidence = response.get("evidence") or {}
    explanation = response.get("recording_explanation") or {}

    beat_results = list(summary.get("beat_results") or [])
    probabilities = list(prediction.get("probabilities") or [])

    # --- derived signals ---------------------------------------------------
    recording_entropy = normalised_entropy(probabilities)
    beat_entropies = _beat_entropies(beat_results)
    high_entropy_beats = sum(1 for e in beat_entropies if e > ENTROPY_CEILING)
    beat_count = summary.get("total_valid_beats") or len(beat_results)
    abnormal_fraction = (
        float(summary.get("abnormal_beat_percentage", 0.0)) / 100.0
        if summary.get("abnormal_beat_percentage") is not None
        else None
    )

    low_reliability_evidence = [
        item for item in (evidence.get("supporting") or [])
        if float(item.get("reliability", 1.0)) < RELIABILITY_FLOOR
    ]

    blind_check = verify_blind_classes(probabilities, class_labels)

    # --- governance gates --------------------------------------------------
    reasons: List[str] = []

    confidence = prediction.get("confidence")
    if confidence is not None and confidence < CONFIDENCE_FLOOR:
        reasons.append(
            f"recording confidence {confidence:.3f} below {CONFIDENCE_FLOOR} floor"
        )

    if recording_entropy is not None and recording_entropy > ENTROPY_CEILING:
        reasons.append(
            f"normalised prediction entropy {recording_entropy:.3f} above "
            f"{ENTROPY_CEILING} ceiling - model is not decisive between classes"
        )

    if abnormal_fraction is not None and abnormal_fraction > ABNORMAL_BURDEN_CEILING:
        reasons.append(
            f"{summary.get('abnormal_beat_count')} of {beat_count} beats abnormal "
            f"({abnormal_fraction:.1%}), above {ABNORMAL_BURDEN_CEILING:.0%} threshold"
        )

    consistency = reasoning.get("consistency_status")
    if consistency and str(consistency).upper() not in ("CONSISTENT", "SUPPORTED"):
        reasons.append(
            f"reasoning engine reported consistency status '{consistency}' - "
            f"model prediction and evidence do not agree"
        )

    if evidence.get("conflicting"):
        reasons.append(
            f"{len(evidence['conflicting'])} conflicting evidence item(s) "
            f"reported by the evidence builder"
        )

    if blind_check.get("verified") and not blind_check.get("assumption_holds"):
        reasons.append(blind_check["note"])

    if narrative.get("fallback_used"):
        reasons.append(
            "narrative generation fell back to a template - the language model "
            "was unavailable, so the written report is not the primary output"
        )

    for source, warns in (
        ("signal", response.get("warnings")),
        ("explanation", explanation.get("warnings")),
        ("narrative", narrative.get("warnings")),
    ):
        if warns:
            reasons.append(f"{len(warns)} warning(s) raised during {source} processing")

    if low_reliability_evidence:
        reasons.append(
            f"{len(low_reliability_evidence)} supporting evidence item(s) below "
            f"reliability {RELIABILITY_FLOOR}"
        )

    return {
        # ---- identity and provenance (api) ----
        "record_id": signal.get("record_id"),
        "duration_seconds": signal.get("duration_seconds"),
        "lead_name": signal.get("lead_name"),
        "model_version": prediction.get("model_version"),
        "checkpoint_hash": prediction.get("checkpoint_hash"),
        "preprocessing_version": prediction.get("preprocessing_version"),
        "evidence_version": evidence.get("evidence_version"),
        "reasoning_version": reasoning.get("reasoning_version"),
        "report_version": report.get("report_version"),

        # ---- clinical content (api, verbatim, never regenerated) ----
        "predicted_label": prediction.get("predicted_label"),
        "prediction_confidence": confidence,
        "dominant_label": summary.get("dominant_predicted_label"),
        "class_counts": summary.get("class_counts"),
        "abnormal_beat_count": summary.get("abnormal_beat_count"),
        "total_valid_beats": beat_count,
        "consistency_status": consistency,
        "reasoning_confidence": reasoning.get("reasoning_confidence"),
        "conclusion": reasoning.get("conclusion"),
        "rule_trace": reasoning.get("rule_trace"),
        "clinical_summary": report.get("summary"),
        "recommended_action": report.get("recommended_action"),
        "supporting_findings": report.get("supporting_findings"),
        "conflicting_findings": report.get("conflicting_findings"),
        "clinical_limitations": report.get("limitations"),
        "doctor_report": narrative.get("doctor_report"),
        "next_of_kin_summary": narrative.get("next_of_kin_summary"),
        "narrative_fallback_used": narrative.get("fallback_used"),

        # ---- derived operational signals ----
        "recording_entropy": recording_entropy,
        "high_entropy_beat_count": high_entropy_beats,
        "abnormal_fraction": abnormal_fraction,
        "low_reliability_evidence_count": len(low_reliability_evidence),
        "blind_class_check": blind_check,

        # ---- governance decision ----
        "human_review_required": bool(reasons),
        "review_reasons": reasons,
        "governance_policy": POLICY_VERSION,
        "route": "Clinical Review Queue" if reasons else "Routine Monitoring",

        "field_provenance": {
            "api": [
                "record_id", "predicted_label", "prediction_confidence",
                "class_counts", "abnormal_beat_count", "total_valid_beats",
                "consistency_status", "reasoning_confidence", "conclusion",
                "rule_trace", "clinical_summary", "recommended_action",
                "supporting_findings", "conflicting_findings",
                "doctor_report", "next_of_kin_summary", "checkpoint_hash",
                "model_version", "preprocessing_version",
            ],
            "derived": [
                "recording_entropy (Shannon entropy of prediction.probabilities, "
                "normalised by log(n_classes))",
                "high_entropy_beat_count (same formula per beat_results entry)",
                "abnormal_fraction (recording_summary.abnormal_beat_percentage / 100)",
                "low_reliability_evidence_count (evidence.supporting filtered by "
                "reliability < policy floor)",
                "blind_class_check (declared blind labels tested against emitted "
                "probability mass)",
            ],
            "governance_only": [
                "human_review_required", "review_reasons", "route",
                "governance_policy", "CONFIDENCE_FLOOR", "ENTROPY_CEILING",
                "ABNORMAL_BURDEN_CEILING", "RELIABILITY_FLOOR",
            ],
            "unavailable_from_api": {
                "true_class / ground_truth_miss":
                    "The API performs inference on raw ECG and does not return "
                    "ground truth. Retrospective accuracy metrics belong to the "
                    "evaluation pipeline, not to runtime governance. The previous "
                    "audit computed these from a labelled cohort file; that is an "
                    "offline evaluation activity and has been removed from the "
                    "runtime path.",
                "calibrated_confidence / expected_calibration_error":
                    "Not exposed by the API. The previous pipeline subtracted a "
                    "fixed ECE from raw confidence; that correction was of "
                    "questionable direction (measured confidence_gap was negative, "
                    "indicating under-confidence) and has been dropped rather than "
                    "carried forward uncorrected. Raw confidence is used directly.",
                "per_class_recall":
                    "Belongs to the evaluation pipeline. If a recall floor gate is "
                    "wanted, the API would need to expose per-class reliability for "
                    "the deployed checkpoint.",
                "mc_dropout_uncertainty_level":
                    "Not exposed. Replaced by entropy over prediction.probabilities, "
                    "which is derived from the deployed model rather than a separate "
                    "offline artefact.",
            },
        },
    }

# ---------------------------------------------------------------------------
# REQUEST CONTRACT
#
# Read from api/routes/analyse.py on origin/api-service (b3f9286).
# The endpoint is multipart/form-data, NOT JSON. n8n's HTTP Request node must
# use Body Content Type = "Form-Data" with an n8n binary property for `file`.
# ---------------------------------------------------------------------------
ENDPOINT_PATH = "/api/v1/analyse"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

# Form fields the endpoint accepts. Only `file` and `sampling_rate_hz` are
# required; `record_id` falls back to the uploaded filename stem.
REQUEST_FIELDS = {
    "file": "binary - one complete, unsegmented 1-D NumPy ECG recording. "
            "Beat segmentation happens server-side. Extension must be in the "
            "service's supported_suffixes (verify with the pipeline team).",
    "sampling_rate_hz": "float > 0, REQUIRED",
    "record_id": "str, optional - defaults to the filename stem",
    "lead_name": "str, optional",
    "include_explanations": "bool, default True",
    "include_overlay": "bool, default True",
    "overlay_start_sample": "int >= 0, optional",
    "overlay_stop_sample": "int > 0, optional",
    "overlay_downsample_limit": "int >= 1, optional",
}


def governance_request_form(
    *,
    sampling_rate_hz: float,
    record_id: Optional[str] = None,
    lead_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Form fields for a governance batch run.

    include_explanations and include_overlay are set FALSE deliberately.

    Governance consumes prediction, recording_summary, evidence, reasoning,
    clinical_report and narrative. It reads nothing from
    recording_explanation or recording_attribution_overlay - those exist for
    the frontend. Leaving both at their default True makes the API build
    per-beat attribution maps and a full-resolution overlay for every
    recording, then serialise them, for data this layer discards.

    On a long batch run that is the difference between a response of a few
    kilobytes and one of many megabytes per recording. Turn them on for a
    single recording a clinician is actually looking at; leave them off for
    the queue.
    """
    form: Dict[str, Any] = {
        "sampling_rate_hz": str(sampling_rate_hz),
        "include_explanations": "false",
        "include_overlay": "false",
    }
    if record_id:
        form["record_id"] = record_id
    if lead_name:
        form["lead_name"] = lead_name
    return form


# Error responses the fail-safe branch must handle. All of these mean the
# recording was NOT analysed - escalate, never treat as a clear result.
ERROR_STATUSES = {
    413: "Upload exceeds the 25 MB limit - recording not analysed",
    415: "Unsupported ECG file format - recording not analysed",
    422: "Malformed request or unprocessable recording - not analysed",
    503: "Pipeline service unavailable - recording not analysed",
}


def failsafe_record(
    status_code: int,
    record_id: Optional[str],
    detail: Optional[str] = None,
) -> Dict[str, Any]:
    """Governance record for a recording the API could not analyse.

    The operational principle from the previous architecture carries over
    unchanged: a recording that cannot be assessed is escalated to a human,
    never silently dropped and never counted as normal.
    """
    reason = ERROR_STATUSES.get(
        status_code, f"API returned HTTP {status_code} - recording not analysed"
    )
    return {
        "record_id": record_id,
        "api_status": "error",
        "api_status_code": status_code,
        "api_detail": detail,
        "human_review_required": True,
        "review_reasons": [f"FAIL-SAFE: {reason}"],
        "route": "Clinical Review Queue",
        "governance_policy": POLICY_VERSION,
        "field_provenance": {"governance_only": ["fail-safe escalation"]},
    }
