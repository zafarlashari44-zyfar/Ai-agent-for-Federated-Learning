"""
profiles_service.py — Simulated patient profiles and continuous monitoring.

WHY THIS EXISTS
---------------
The supervisor asked for user profiles under continuous monitoring, of the kind
a wearable or bedside monitor would produce, so the system can be seen watching
someone over time rather than scoring one beat in isolation.

WHAT IS REAL AND WHAT IS SIMULATED — read this before demoing
--------------------------------------------------------------
REAL:
  - Every beat assigned to a profile is a genuine MIT-BIH record drawn from the
    project's own cohort file, with its true label, model prediction, model
    confidence and MC-Dropout uncertainty exactly as the FL pipeline produced.
  - The per-class reliability figures used to flag risk come from the team's
    own evaluation artefacts.

SIMULATED:
  - The patients themselves. Names, ages and admission context are invented.
  - The assignment of beats to a patient, and the ordering of those beats into
    a monitoring timeline. MIT-BIH beats are not from these people and carry no
    real temporal relationship to one another.
  - The wall-clock timestamps on the stream.

WHAT THIS DOES *NOT* DO
-----------------------
It does not predict cardiac arrest. A per-beat classifier trained on beat-level
AAMI labels cannot forecast a future event, and claiming otherwise would not
survive scrutiny. What the deterioration signal below actually reports is a
rising density of abnormal and unreliable classifications within a sliding
window — an evidence-accumulation trigger for human review, nothing more. It is
labelled that way throughout, in the code and in the API responses.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

# Classes the federated model cannot emit (recall 0.0 in the team's own
# per-class evaluation). Kept here so the monitoring layer can flag a beat that
# the model is structurally incapable of detecting.
MODEL_BLIND_CLASSES = {"S", "F"}

# Sliding-window trigger for review. Deliberately conservative and deliberately
# NOT called a prediction.
#
# Thresholds are NOT hardcoded guesses. They are derived at runtime from the
# cohort's own base rates, following the same principle as the governance layer:
# every threshold traces to a measured property of the data. A window fires only
# when it is materially worse than this cohort's normal state. A fixed threshold
# would fire for every profile at once - because roughly 27% of this cohort is
# already of an undetectable class - which would make the signal useless.
WINDOW_SIZE = 10
EXCESS_MARGIN = 1.5  # window rate must exceed base rate by this multiple


# --------------------------------------------------------------------------
# Profile definitions — simulated demographics, real beats
# --------------------------------------------------------------------------
PROFILE_SPECS: List[Dict[str, Any]] = [
    {"profile_id": "USR-001", "display_name": "Simulated Patient A", "age": 67,
     "sex": "F", "context": "Post-MI telemetry, day 3", "device": "Simulated 3-lead telemetry"},
    {"profile_id": "USR-002", "display_name": "Simulated Patient B", "age": 54,
     "sex": "M", "context": "Palpitations, ambulatory Holter", "device": "Simulated Holter monitor"},
    {"profile_id": "USR-003", "display_name": "Simulated Patient C", "age": 72,
     "sex": "M", "context": "Known AF on rate control", "device": "Simulated bedside monitor"},
    {"profile_id": "USR-004", "display_name": "Simulated Patient D", "age": 41,
     "sex": "F", "context": "Syncope investigation", "device": "Simulated patch monitor"},
    {"profile_id": "USR-005", "display_name": "Simulated Patient E", "age": 79,
     "sex": "F", "context": "Heart failure, remote monitoring", "device": "Simulated wearable"},
    {"profile_id": "USR-006", "display_name": "Simulated Patient F", "age": 33,
     "sex": "M", "context": "Athlete screening, incidental ectopy", "device": "Simulated chest strap"},
]


def _cohort() -> Dict[str, Any]:
    """Import lazily from persona_service so both modules share one cached read."""
    from persona_service import load_cohort
    return load_cohort()


def _stable_bucket(sample_index: Any, n_buckets: int) -> int:
    """Deterministic assignment of a beat to a profile.

    Hashing the sample index rather than using round-robin or random.shuffle
    means the same beat lands with the same patient on every run, on every
    machine, with no seed to remember. The demo is reproducible.
    """
    digest = hashlib.md5(str(sample_index).encode()).hexdigest()
    return int(digest, 16) % n_buckets


@lru_cache(maxsize=1)
def build_profiles() -> Dict[str, Dict[str, Any]]:
    """Distribute real cohort beats across the simulated profiles."""
    records = _cohort().get("records", [])
    profiles: Dict[str, Dict[str, Any]] = {}

    for spec in PROFILE_SPECS:
        profiles[spec["profile_id"]] = {**spec, "beats": []}

    ids = [s["profile_id"] for s in PROFILE_SPECS]
    for rec in records:
        pid = ids[_stable_bucket(rec.get("sample_index"), len(ids))]
        profiles[pid]["beats"].append(rec)

    # Order each profile's beats by a second independent hash rather than
    # leaving them in cohort order. The cohort file may be grouped by class,
    # which would put long runs of one class at the end of every timeline and
    # make the sliding window fire for every profile at once. Hashing gives a
    # deterministic but class-mixed ordering, so the monitoring signal
    # discriminates between profiles instead of saturating.
    for prof in profiles.values():
        prof["beats"].sort(
            key=lambda b: hashlib.md5(f"order-{b.get('sample_index')}".encode()).hexdigest()
        )

    # Attach a synthetic monitoring timeline: one beat per simulated minute,
    # counting back from now so the most recent beat is "just now".
    now = datetime.now(timezone.utc)
    for prof in profiles.values():
        n = len(prof["beats"])
        for i, beat in enumerate(prof["beats"]):
            beat["_monitor_ts"] = (now - timedelta(minutes=(n - i))).isoformat()

    return profiles


@lru_cache(maxsize=1)
def cohort_base_rates() -> Dict[str, float]:
    """Base rates across the whole cohort, used to calibrate the trigger."""
    records = _cohort().get("records", [])
    if not records:
        return {"abnormal": 0.0, "blind": 0.0, "uncertain": 0.0}
    n = len(records)
    flags = [_beat_flags(r) for r in records]
    return {
        "abnormal": sum(f["abnormal_prediction"] for f in flags) / n,
        "blind": sum(f["model_blind_true_class"] for f in flags) / n,
        "uncertain": sum(f["high_uncertainty"] for f in flags) / n,
    }


def _beat_flags(beat: Dict[str, Any]) -> Dict[str, Any]:
    """Per-beat risk signals, all traceable to pipeline output."""
    true_cls = beat.get("true_class_aami") or beat.get("true_label")
    pred = str(beat.get("prediction", ""))
    abnormal = "normal" not in pred.lower()
    return {
        "abnormal_prediction": abnormal,
        "high_uncertainty": str(beat.get("uncertainty_level", "")).lower() == "high",
        "model_blind_true_class": true_cls in MODEL_BLIND_CLASSES,
        "misclassified": beat.get("correct") in (0, False),
    }


def summarise_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    beats = profile["beats"]
    if not beats:
        return {"beat_count": 0}

    flags = [_beat_flags(b) for b in beats]
    n = len(beats)
    confs = [b.get("confidence") for b in beats if b.get("confidence") is not None]

    return {
        "beat_count": n,
        "abnormal_predictions": sum(f["abnormal_prediction"] for f in flags),
        "high_uncertainty_beats": sum(f["high_uncertainty"] for f in flags),
        "beats_of_undetectable_class": sum(f["model_blind_true_class"] for f in flags),
        "misclassified_beats": sum(f["misclassified"] for f in flags),
        "mean_confidence": round(sum(confs) / len(confs), 4) if confs else None,
        "monitoring_window_start": beats[0].get("_monitor_ts"),
        "monitoring_window_end": beats[-1].get("_monitor_ts"),
    }


def deterioration_signal(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Sliding-window evidence accumulation over the most recent beats.

    NOT a cardiac-arrest predictor. This reports that abnormal or undetectable
    beats have become dense enough in the recent window, RELATIVE TO THIS
    COHORT'S OWN BASE RATE, to warrant a human looking at the trace.
    """
    beats = profile["beats"]
    window = beats[-WINDOW_SIZE:]
    if not window:
        return {"status": "insufficient_data", "beats_in_window": 0}

    base = cohort_base_rates()
    flags = [_beat_flags(b) for b in window]
    n = len(window)

    rates = {
        "abnormal": sum(f["abnormal_prediction"] for f in flags) / n,
        "blind": sum(f["model_blind_true_class"] for f in flags) / n,
        "uncertain": sum(f["high_uncertainty"] for f in flags) / n,
    }

    reasons: List[str] = []
    if rates["abnormal"] > base["abnormal"] * EXCESS_MARGIN and rates["abnormal"] > 0:
        reasons.append(
            f"abnormal classifications in the last {n} beats ran at "
            f"{rates['abnormal']:.0%} against a cohort base rate of {base['abnormal']:.0%}"
        )
    if rates["blind"] > base["blind"] * EXCESS_MARGIN and rates["blind"] > 0:
        reasons.append(
            f"{int(rates['blind'] * n)} of {n} recent beats have a true class the model "
            f"cannot detect (recall 0.0 for S and F), against a cohort base rate of "
            f"{base['blind']:.0%} - these beats are invisible to the classifier"
        )
    if rates["uncertain"] > base["uncertain"] * EXCESS_MARGIN and rates["uncertain"] > 0:
        reasons.append(
            f"{int(rates['uncertain'] * n)} of {n} recent beats fell in the high "
            f"MC-Dropout uncertainty band, where measured accuracy is 66.4% rather "
            f"than 96.8%"
        )

    return {
        "status": "review_recommended" if reasons else "no_trigger",
        "beats_in_window": n,
        "window_rates": {k: round(v, 3) for k, v in rates.items()},
        "cohort_base_rates": {k: round(v, 3) for k, v in base.items()},
        "excess_margin": EXCESS_MARGIN,
        "reasons": reasons,
        "interpretation": (
            "Evidence-accumulation trigger for human review over a sliding window of "
            "recent classifications, calibrated against this cohort's own base rates. "
            "This is NOT a prediction of a future cardiac event; the underlying model "
            "classifies individual beats and has no forecasting capability."
        ),
    }


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
class ProfileSummary(BaseModel):
    profile_id: str
    display_name: str
    age: int
    sex: str
    context: str
    device: str
    simulated: bool = True
    summary: Dict[str, Any]
    monitoring: Dict[str, Any]


@router.get("/api/profiles")
def list_profiles():
    """All simulated profiles with their monitoring status."""
    out = []
    for prof in build_profiles().values():
        out.append({
            "profile_id": prof["profile_id"],
            "display_name": prof["display_name"],
            "age": prof["age"],
            "sex": prof["sex"],
            "context": prof["context"],
            "device": prof["device"],
            "simulated": True,
            "summary": summarise_profile(prof),
            "monitoring": deterioration_signal(prof),
        })
    return {
        "profiles": out,
        "disclosure": (
            "Patient identities, ages, clinical context and monitoring timestamps are "
            "SIMULATED. The ECG beats, model predictions, confidences and uncertainty "
            "values attached to each profile are real records from the project's MIT-BIH "
            "cohort."
        ),
    }


@router.get("/api/profiles/{profile_id}")
def get_profile(profile_id: str, limit: int = 25):
    profiles = build_profiles()
    pid = profile_id.strip().upper()
    if pid not in profiles:
        raise HTTPException(status_code=404, detail=f"No profile '{profile_id}'.")
    prof = profiles[pid]
    beats = prof["beats"][-limit:]
    return {
        "profile": {k: v for k, v in prof.items() if k != "beats"},
        "simulated": True,
        "summary": summarise_profile(prof),
        "monitoring": deterioration_signal(prof),
        "recent_beats": [
            {
                "monitor_timestamp": b.get("_monitor_ts"),
                "patient_id": b.get("patient_id"),
                "sample_index": b.get("sample_index"),
                "prediction": b.get("prediction"),
                "true_class_aami": b.get("true_class_aami") or b.get("true_label"),
                "confidence": b.get("confidence"),
                "uncertainty_level": b.get("uncertainty_level"),
                "flags": _beat_flags(b),
            }
            for b in beats
        ],
    }


@router.get("/api/profiles/{profile_id}/stream")
def stream_window(profile_id: str, window: int = WINDOW_SIZE):
    """The most recent window and its trigger state — what a monitoring
    dashboard would poll."""
    profiles = build_profiles()
    pid = profile_id.strip().upper()
    if pid not in profiles:
        raise HTTPException(status_code=404, detail=f"No profile '{profile_id}'.")
    prof = profiles[pid]
    return {
        "profile_id": pid,
        "display_name": prof["display_name"],
        "simulated": True,
        "monitoring": deterioration_signal(prof),
        "window": [
            {
                "monitor_timestamp": b.get("_monitor_ts"),
                "prediction": b.get("prediction"),
                "confidence": b.get("confidence"),
                "uncertainty_level": b.get("uncertainty_level"),
                "true_class_aami": b.get("true_class_aami") or b.get("true_label"),
            }
            for b in prof["beats"][-window:]
        ],
    }
