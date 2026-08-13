"""
analyse_adapter.py — maps the ECG Reasoning Pipeline's /api/v1/analyse
response onto the fields the n8n governance layer needs.

Written against origin/api-service @ 1356dfe ("Add signal suitability and
heuristic OOD assessment"). All field names and enum values are taken from
api/schemas/analyse.py and domain/enums/statuses.py. Nothing is guessed.

RESPONSIBILITY BOUNDARY
-----------------------
The Reasoning Pipeline is the single source of clinical truth. This module
performs NO clinical interpretation, reasoning, narrative generation or
explainability. It reads the API response and derives only the operational
signals governance needs for routing.

GATE ORDER MATTERS
------------------
Gates run in a fixed order and short-circuit. A confidence value computed on
an unsuitable or out-of-distribution signal is not meaningful, so such a
recording must never reach confidence-based routing. The order is:

    1. input_accepted              - was the upload usable at all
    2. signal_suitability          - is the signal processable
    3. ood_assessment              - is it the kind of data the model knows
    4. analysis_scope              - is the result within validated scope
    5. model_prediction_produced   - did inference actually run
    6. reasoning consistency       - does evidence agree with the prediction
    7. confidence / burden         - only now does risk routing apply

REMOVED IN THIS VERSION
-----------------------
The previous adapter derived normalised prediction entropy from the
probability tuple. The API now returns
`ood_assessment.normalized_prediction_entropy` directly, computed by the
pipeline that owns the model. The local derivation has been deleted rather
than kept as a parallel implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# ENUM VALUES — copied from domain/enums/statuses.py @ 1356dfe
# ---------------------------------------------------------------------------
SUITABILITY_ACCEPTED = "accepted"
SUITABILITY_ACCEPTED_WITH_WARNINGS = "accepted_with_warnings"
SUITABILITY_REJECTED = "rejected"

OOD_IN_DISTRIBUTION_LIKE = "in_distribution_like"
OOD_UNCERTAIN = "uncertain"
OOD_LIKELY_OUT_OF_DISTRIBUTION = "likely_out_of_distribution"

SCOPE_VALIDATED = "validated_mit_bih_compatible"
SCOPE_EXPLORATORY = "exploratory_external_source"
SCOPE_UNSUPPORTED = "unsupported"

CONSISTENCY_STRONGLY_SUPPORTED = "strongly_supported"
CONSISTENCY_PARTIALLY_SUPPORTED = "partially_supported"
CONSISTENCY_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
CONSISTENCY_CONFLICTING_EVIDENCE = "conflicting_evidence"
CONSISTENCY_LOW_SIGNAL_QUALITY = "low_signal_quality"
CONSISTENCY_OUT_OF_SCOPE = "out_of_scope"

# Only this one clears without comment. PARTIALLY_SUPPORTED is deliberately
# NOT auto-cleared: partial support means the evidence builder could not fully
# corroborate the prediction, which is exactly when a human should look.
CONSISTENCY_CLEARS = (CONSISTENCY_STRONGLY_SUPPORTED,)

# ---------------------------------------------------------------------------
# GOVERNANCE POLICY
#
# The only numbers this layer owns. Policy, not clinical fact, and stated here
# rather than buried in a workflow node so a reviewer can find and challenge
# them. PROVISIONAL until calibrated against real API responses.
# ---------------------------------------------------------------------------
POLICY_VERSION = "governance-4.0-api-1356dfe"

CONFIDENCE_FLOOR = 0.70
ABNORMAL_BURDEN_CEILING = 0.20
RELIABILITY_FLOOR = 0.50
QUALITY_SCORE_FLOOR = 0.60
ENTROPY_CEILING = 0.55

ENDPOINT_PATH = "/api/v1/analyse"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


# ---------------------------------------------------------------------------
# Request side
# ---------------------------------------------------------------------------
REQUEST_FIELDS = {
    "file": "binary - one complete, unsegmented 1-D ECG recording. Beat "
            "segmentation happens server-side.",
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
    Governance reads nothing from recording_explanation or
    recording_attribution_overlay - those exist for the frontend. Leaving the
    defaults on makes the pipeline compute and serialise per-beat attribution
    maps for every recording, for data this layer discards.

    Turn them on for a single recording a clinician is reviewing. Leave them
    off for the queue.
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

    A recording that cannot be assessed is escalated to a human, never
    silently dropped and never counted as normal.
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
        "gate_triggered": "transport",
        "review_reasons": [f"FAIL-SAFE: {reason}"],
        "route": "Clinical Review Queue",
        "governance_policy": POLICY_VERSION,
    }


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def _gate_input_accepted(r: Dict[str, Any]) -> Optional[str]:
    if r.get("input_accepted") is False:
        return ("input not accepted by the pipeline - the upload could not be "
                "ingested, so no clinical result exists for this recording")
    return None


def _gate_suitability(r: Dict[str, Any]) -> Optional[str]:
    suit = r.get("signal_suitability")
    if suit is None:
        return ("signal suitability not assessed - cannot confirm the recording "
                "was processable")

    status = suit.get("status")
    if status == SUITABILITY_REJECTED:
        rejections = "; ".join(suit.get("rejection_reasons") or []) or "no reason given"
        return f"signal REJECTED by suitability check: {rejections}"

    if status == SUITABILITY_ACCEPTED_WITH_WARNINGS:
        warns = "; ".join(suit.get("warnings") or []) or "no detail given"
        return f"signal accepted with warnings: {warns}"

    quality = suit.get("quality_score")
    if quality is not None and quality < QUALITY_SCORE_FLOOR:
        return (f"signal quality score {quality:.2f} below {QUALITY_SCORE_FLOOR} "
                f"floor (flatline {suit.get('flatline_percentage')}%, "
                f"clipping {suit.get('clipping_percentage')}%, "
                f"noise {suit.get('noise_score')})")
    return None


def _gate_ood(r: Dict[str, Any]) -> Optional[str]:
    ood = r.get("ood_assessment")
    if ood is None:
        return "out-of-distribution assessment absent - cannot confirm the recording resembles training data"

    status = ood.get("status")
    if status == OOD_LIKELY_OUT_OF_DISTRIBUTION:
        reasons = "; ".join(ood.get("reasons") or []) or "no reason given"
        return (f"recording assessed as LIKELY OUT OF DISTRIBUTION "
                f"(heuristic score {ood.get('heuristic_score')}): {reasons}")

    if status == OOD_UNCERTAIN:
        indicators = "; ".join(ood.get("indicators") or []) or "no indicators listed"
        return f"out-of-distribution status UNCERTAIN: {indicators}"

    entropy = ood.get("normalized_prediction_entropy")
    if entropy is not None and entropy > ENTROPY_CEILING:
        return (f"normalised prediction entropy {entropy:.3f} above "
                f"{ENTROPY_CEILING} - the model is not decisive between classes")
    return None


def _gate_scope(r: Dict[str, Any]) -> Optional[str]:
    scope = r.get("analysis_scope")
    if scope == SCOPE_UNSUPPORTED:
        return ("analysis scope UNSUPPORTED - this recording is outside what "
                "the model has been validated for")
    if scope == SCOPE_EXPLORATORY:
        return ("analysis scope EXPLORATORY (external source) - the model was "
                "validated on MIT-BIH-compatible data, so this result is "
                "indicative only and must not be acted on unreviewed")
    return None


def _gate_prediction_produced(r: Dict[str, Any]) -> Optional[str]:
    if r.get("model_prediction_produced") is False:
        return ("no model prediction was produced - the pipeline ran but "
                "inference did not complete")
    return None


def _gate_consistency(r: Dict[str, Any]) -> Optional[str]:
    reasoning = r.get("reasoning") or {}
    status = reasoning.get("consistency_status")
    if status is None:
        return None
    if status in CONSISTENCY_CLEARS:
        return None
    labels = {
        CONSISTENCY_PARTIALLY_SUPPORTED:
            "evidence only PARTIALLY supports the prediction",
        CONSISTENCY_INSUFFICIENT_EVIDENCE:
            "INSUFFICIENT evidence to corroborate the prediction",
        CONSISTENCY_CONFLICTING_EVIDENCE:
            "evidence CONFLICTS with the prediction",
        CONSISTENCY_LOW_SIGNAL_QUALITY:
            "reasoning engine flagged LOW SIGNAL QUALITY",
        CONSISTENCY_OUT_OF_SCOPE:
            "reasoning engine flagged the recording OUT OF SCOPE",
    }
    return f"reasoning consistency: {labels.get(status, status)}"


def _gate_risk(r: Dict[str, Any]) -> List[str]:
    """Confidence and burden. Only reached if every earlier gate passed."""
    out: List[str] = []
    prediction = r.get("prediction") or {}
    summary = r.get("recording_summary") or {}
    evidence = r.get("evidence") or {}
    narrative = r.get("narrative") or {}

    confidence = prediction.get("confidence")
    if confidence is not None and confidence < CONFIDENCE_FLOOR:
        out.append(f"prediction confidence {confidence:.3f} below {CONFIDENCE_FLOOR} floor")

    pct = summary.get("abnormal_beat_percentage")
    if pct is not None and (pct / 100.0) > ABNORMAL_BURDEN_CEILING:
        out.append(
            f"{summary.get('abnormal_beat_count')} of "
            f"{summary.get('total_valid_beats')} beats abnormal ({pct:.1f}%), "
            f"above {ABNORMAL_BURDEN_CEILING:.0%} threshold"
        )

    conflicting = evidence.get("conflicting") or []
    if conflicting:
        out.append(f"{len(conflicting)} conflicting evidence item(s)")

    weak = [
        i for i in (evidence.get("supporting") or [])
        if float(i.get("reliability", 1.0)) < RELIABILITY_FLOOR
    ]
    if weak:
        out.append(
            f"{len(weak)} supporting evidence item(s) below reliability {RELIABILITY_FLOOR}"
        )

    if narrative.get("fallback_used"):
        out.append(
            "narrative generation fell back to a template - the language model "
            "was unavailable, so the written report is not the primary output"
        )
    return out


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------
GATE_SEQUENCE = (
    ("input_accepted", _gate_input_accepted),
    ("signal_suitability", _gate_suitability),
    ("ood_assessment", _gate_ood),
    ("analysis_scope", _gate_scope),
    ("model_prediction_produced", _gate_prediction_produced),
    ("reasoning_consistency", _gate_consistency),
)


def adapt(response: Dict[str, Any]) -> Dict[str, Any]:
    """Map one /api/v1/analyse response onto governance inputs.

    Early gates short-circuit: the first that fires stops evaluation, because
    downstream signals are not meaningful once an upstream one has failed.
    `gate_triggered` records which one, so escalations can be counted by cause
    rather than lumped together.
    """
    signal = response.get("signal") or {}
    prediction = response.get("prediction") or {}
    summary = response.get("recording_summary") or {}
    reasoning = response.get("reasoning") or {}
    report = response.get("clinical_report") or {}
    narrative = response.get("narrative") or {}
    evidence = response.get("evidence") or {}
    suitability = response.get("signal_suitability") or {}
    ood = response.get("ood_assessment") or {}

    reasons: List[str] = []
    gate_triggered: Optional[str] = None

    for name, gate in GATE_SEQUENCE:
        result = gate(response)
        if result:
            gate_triggered = name
            reasons.append(result)
            break

    # Risk routing only runs when every upstream gate passed.
    if gate_triggered is None:
        risk_reasons = _gate_risk(response)
        if risk_reasons:
            gate_triggered = "risk_routing"
            reasons.extend(risk_reasons)

    return {
        # ---- identity and provenance ----
        "record_id": signal.get("record_id"),
        "duration_seconds": signal.get("duration_seconds"),
        "lead_name": signal.get("lead_name"),
        "model_version": prediction.get("model_version"),
        "checkpoint_hash": prediction.get("checkpoint_hash"),
        "preprocessing_version": prediction.get("preprocessing_version"),
        "evidence_version": evidence.get("evidence_version"),
        "reasoning_version": reasoning.get("reasoning_version"),
        "report_version": report.get("report_version"),

        # ---- upstream assessment (api) ----
        "input_accepted": response.get("input_accepted"),
        "model_prediction_produced": response.get("model_prediction_produced"),
        "analysis_scope": response.get("analysis_scope"),
        "model_scope_statement": response.get("model_scope_statement"),
        "suitability_status": suitability.get("status"),
        "suitability_quality_score": suitability.get("quality_score"),
        "suitability_rejection_reasons": suitability.get("rejection_reasons"),
        "detected_r_peak_count": suitability.get("detected_r_peak_count"),
        "estimated_heart_rate_bpm": suitability.get("estimated_heart_rate_bpm"),
        "ood_status": ood.get("status"),
        "ood_heuristic_score": ood.get("heuristic_score"),
        "ood_normalized_entropy": ood.get("normalized_prediction_entropy"),
        "ood_indicators": ood.get("indicators"),

        # ---- clinical content (api, verbatim, never regenerated) ----
        "predicted_label": prediction.get("predicted_label"),
        "prediction_confidence": prediction.get("confidence"),
        "dominant_label": summary.get("dominant_predicted_label"),
        "class_counts": summary.get("class_counts"),
        "abnormal_beat_count": summary.get("abnormal_beat_count"),
        "abnormal_beat_percentage": summary.get("abnormal_beat_percentage"),
        "total_valid_beats": summary.get("total_valid_beats"),
        "consistency_status": reasoning.get("consistency_status"),
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

        # ---- governance decision ----
        "human_review_required": bool(reasons),
        "gate_triggered": gate_triggered,
        "review_reasons": reasons,
        "governance_policy": POLICY_VERSION,
        "route": "Clinical Review Queue" if reasons else "Routine Monitoring",

        "field_provenance": {
            "api": "everything above except the governance block",
            "derived": "none - the previous local entropy derivation was "
                       "removed in favour of "
                       "ood_assessment.normalized_prediction_entropy",
            "governance_only": [
                "human_review_required", "gate_triggered", "review_reasons",
                "route", "governance_policy",
                "CONFIDENCE_FLOOR", "ABNORMAL_BURDEN_CEILING",
                "RELIABILITY_FLOOR", "QUALITY_SCORE_FLOOR", "ENTROPY_CEILING",
            ],
            "unavailable_from_api": {
                "calibrated_confidence":
                    "Not exposed. Agreed with the pipeline team: no ECE "
                    "correction is applied in n8n. Raw confidence is used and "
                    "n8n owns the operational thresholds.",
                "ground_truth / accuracy":
                    "The API performs inference and returns no labels. "
                    "Retrospective accuracy is an evaluation activity, not "
                    "runtime governance.",
                "per_class_recall":
                    "Belongs to the evaluation pipeline. A per-class "
                    "reliability field on PredictionResponse would let the "
                    "blind-class concern be expressed as a gate.",
            },
        },
    }
