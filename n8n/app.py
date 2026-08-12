"""
Agentic Federated Learning — Clinical Intelligence Service (v2)

Changes from v1, and why each one matters for the orchestration layer:

1. PatientPayload now accepts the full 23-column feature set instead of 5.
   v1 hardcoded confidence=0.0, rr_category="Unknown", risk_flag="Unknown",
   ecg_lead="N/A". Those values were then interpolated into the LLM prompt,
   so every request through the API told the model "Confidence: 0.0%" and
   "Rhythm Context: Unknown". The batch path in clinical_agent.py sent the
   real values. The two paths were therefore NOT comparable — an internal
   validity problem if the dissertation reports results from either.

2. The response echoes the input features back alongside agent_output, so the
   orchestrator has the evidence it needs to apply governance rules without
   re-joining against the source CSV.

3. `status` is renamed in intent: it still reports parse success/failure, but
   a separate `clinical_status` field now carries the Normal/Abnormal call.
   In v1 the n8n IF node compared `status` to "Abnormal" — a value the API
   never emitted — so 100% of patients silently took the Normal branch.

4. NaN-safe serialisation. json.dump writes bare NaN, which is invalid JSON
   per RFC 8259 and is rejected by JavaScript's JSON.parse (and therefore by
   n8n). Optional floats are normalised to None.

All new payload fields are Optional, so v1 clients keep working.

v3 additions
------------
5. The persona router is mounted at the bottom of this file. It adds an
   INTERACTIVE path (/api/ask, /api/patient/{id}, /api/cohort/summary,
   /api/personas) alongside the existing BATCH path (/api/analyze-patient).
   The batch contract is unchanged, so the n8n workflow is unaffected.
"""

import json
import math
from typing import Optional, Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from clinical_agent import generate_agent_response

app = FastAPI(
    title="Agentic Federated Learning - Clinical Intelligence Service",
    description="FastAPI bridge connecting n8n orchestration nodes to local Bio-BERT & Medichat-Llama3 pipelines.",
    version="4.0.0",
)

NORMAL_CLASS = "Normal Sinus Rhythm"


class PatientPayload(BaseModel):
    # --- required identity & prediction ---
    patient_id: str
    prediction: str

    # --- SHAP evidence (required, these drive the clinician briefing) ---
    qrs_shap_signed: float
    rr_variance_proxy: float
    ehr_triage_note: str

    # --- optional: full feature set, defaults preserve v1 compatibility ---
    sample_id: Optional[int] = None
    ecg_lead: str = "Unknown"
    true_label: str = "Unknown"
    confidence: float = 0.0
    correct: Optional[int] = None
    risk_flag: str = "Unknown"

    p_wave_shap_signed: float = 0.0
    t_wave_shap_signed: float = 0.0
    qrs_importance: float = 0.0
    p_wave_importance: float = 0.0
    t_wave_importance: float = 0.0

    dominant_ecg_region: str = "Unknown"
    qrs_direction: str = "Unknown"
    p_wave_direction: str = "Unknown"
    t_wave_direction: str = "Unknown"

    rr_category: str = "Unknown"
    p_wave_present: int = 0
    t_wave_amplitude: Optional[float] = None
    st_elevation: Optional[float] = None
    st_flag: str = "Unknown"

    # --- orchestration metadata passed through from n8n ---
    run_id: Optional[str] = None
    ingested_at: Optional[str] = None


def _nan_safe(value: Optional[float]) -> Optional[float]:
    """Convert NaN/inf to None so the response is valid RFC 8259 JSON."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _fallback_output(payload: PatientPayload, raw: str) -> Dict[str, Any]:
    """Deterministic degraded response when the quantized model breaks JSON sync."""
    abnormal = payload.prediction != NORMAL_CLASS
    return {
        "urgency_level": "High" if abnormal else "Low",
        "suggested_next_action": "Immediate clinician consult" if abnormal else "Routine follow-up",
        "doctor_technical_alert": f"[FALLBACK — model output was not valid JSON] Raw analysis: {raw}",
        "family_reassurance_message": "The medical team is actively reviewing your relative's chart status.",
    }


@app.get("/")
def read_root():
    return {"status": "operational", "service": "Clinical Agent API Bridge", "version": "4.0.0"}


@app.get("/health")
def health():
    """Lightweight probe so the orchestrator can verify the service before a batch run."""
    return {"status": "ok"}


@app.post("/api/analyze-patient")
def analyze_patient(payload: PatientPayload):
    """
    Receives cross-modal patient parameters from n8n, passes them to the local LLM,
    and returns a flat, orchestrator-friendly JSON contract.
    """
    print(f"\n[API] Ingesting record for Patient: {payload.patient_id} ({payload.prediction})")
    raw_llm_response = ""
    parse_status = "success"

    try:
        raw_llm_response = generate_agent_response(
            patient_id=payload.patient_id,
            ecg_lead=payload.ecg_lead,
            true_label=payload.true_label,
            prediction=payload.prediction,
            confidence=payload.confidence,
            correct=bool(payload.correct) if payload.correct is not None else False,
            risk_flag=payload.risk_flag,
            qrs_shap_signed=payload.qrs_shap_signed,
            p_wave_shap_signed=payload.p_wave_shap_signed,
            t_wave_shap_signed=payload.t_wave_shap_signed,
            qrs_importance=payload.qrs_importance,
            p_wave_importance=payload.p_wave_importance,
            t_wave_importance=payload.t_wave_importance,
            dominant_ecg_region=payload.dominant_ecg_region,
            qrs_direction=payload.qrs_direction,
            p_wave_direction=payload.p_wave_direction,
            t_wave_direction=payload.t_wave_direction,
            rr_variance_proxy=payload.rr_variance_proxy,
            rr_category=payload.rr_category,
            p_wave_present=payload.p_wave_present,
            t_wave_amplitude=_nan_safe(payload.t_wave_amplitude),
            st_elevation=_nan_safe(payload.st_elevation),
            st_flag=payload.st_flag,
            ehr_triage_note=payload.ehr_triage_note,
        )

        cleaned = raw_llm_response.replace("```json", "").replace("```", "").strip()
        agent_output = json.loads(cleaned)

    except json.JSONDecodeError:
        print(f"[WARNING] Patient {payload.patient_id} failed JSON serialization. Activating defensive fallback.")
        parse_status = "parse_error"
        agent_output = _fallback_output(payload, raw_llm_response)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    clinical_status = "Normal" if payload.prediction == NORMAL_CLASS else "Abnormal"

    return {
        "patient_id": payload.patient_id,
        "run_id": payload.run_id,
        # Technical outcome of the LLM call
        "status": parse_status,
        # Clinical outcome — this is what the orchestrator routes on
        "clinical_status": clinical_status,
        # Echo the evidence back so n8n can govern without re-joining the CSV
        "features": {
            "prediction": payload.prediction,
            "true_label": payload.true_label,
            "confidence": payload.confidence,
            "risk_flag": payload.risk_flag,
            "ecg_lead": payload.ecg_lead,
            "dominant_ecg_region": payload.dominant_ecg_region,
            "qrs_shap_signed": payload.qrs_shap_signed,
            "p_wave_shap_signed": payload.p_wave_shap_signed,
            "t_wave_shap_signed": payload.t_wave_shap_signed,
            "rr_variance_proxy": payload.rr_variance_proxy,
            "rr_category": payload.rr_category,
            "st_flag": payload.st_flag,
            "t_wave_amplitude": _nan_safe(payload.t_wave_amplitude),
            "st_elevation": _nan_safe(payload.st_elevation),
            "ehr_triage_note": payload.ehr_triage_note,
        },
        "agent_output": agent_output,
    }


# ---------------------------------------------------------------------------
# Persona layer (v3)
#
# Mounted last so that a missing or broken persona_service.py degrades to a
# warning rather than taking the batch endpoint down with it — the n8n
# orchestration run must never fail because the interactive layer is absent.
# ---------------------------------------------------------------------------
try:
    from persona_service import router as persona_router

    app.include_router(persona_router)
    print("[API] Persona layer mounted: /api/ask, /api/patient/{id}, "
          "/api/cohort/summary, /api/personas")
except Exception as _persona_exc:  # noqa: BLE001
    print(f"[API] WARNING - persona layer not mounted ({_persona_exc}). "
          f"Batch endpoint /api/analyze-patient is unaffected.")

try:
    from profiles_service import router as profiles_router

    app.include_router(profiles_router)
    print("[API] Profiles layer mounted: /api/profiles, /api/profiles/{id}, "
          "/api/profiles/{id}/stream")
except Exception as _profiles_exc:  # noqa: BLE001
    print(f"[API] WARNING - profiles layer not mounted ({_profiles_exc}). "
          f"Batch endpoint /api/analyze-patient is unaffected.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
