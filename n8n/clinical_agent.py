import ollama
import json
import pandas as pd
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Core LLM function - UPDATED FOR MULTI-MODAL EHR INGESTION
#
# CHANGE LOG (this revision):
#   1. The SHAP attribution values and physiological features are now
#      interpolated into user_prompt. Previously they were accepted as
#      parameters but never reached the model, while the system instruction
#      told the model to reason from them. Any SHAP reasoning in the output
#      was therefore ungrounded.
#   2. true_label, correct and risk_flag are deliberately NOT placed in the
#      prompt. The model must not see ground truth when producing a clinical
#      briefing. They remain parameters because the batch runner logs them.
#   3. num_predict caps generation length. This model continues past the JSON
#      object if left uncapped, which dominated per-request latency.
# ---------------------------------------------------------------------------

def generate_agent_response(
        patient_id, ecg_lead,
        true_label, prediction, confidence, correct, risk_flag,
        qrs_shap_signed, p_wave_shap_signed, t_wave_shap_signed,
        qrs_importance, p_wave_importance, t_wave_importance,
        dominant_ecg_region,
        qrs_direction, p_wave_direction, t_wave_direction,
        rr_variance_proxy, rr_category,
        p_wave_present, t_wave_amplitude, st_elevation, st_flag,
        ehr_triage_note
):
    system_instruction = """
You are a highly capable medical AI acting as a communication bridge between a diagnostic model, doctors, and patients' families.

RULES:
1. DOCTOR ALERT: Act as a Cardiologist. Explain the clinical reasoning behind the diagnosis using the provided SHAP values, signed directions, and physiological features.
2. FAMILY MESSAGE: Act as an empathetic nurse. Explain the condition at a 6th-grade reading level. ABSOLUTELY NO MEDICAL JARGON. Use simple, everyday analogies. Focus on active monitoring.
3. STRICT GROUNDING FACT-CHECK: You are under penalty of failure if you invent any secondary conditions, congenital syndromes, or diagnostic names (such as Wolff-Parkinson-White or Bundle Branch Blocks) that are not explicitly confirmed in the provided Electronic Health Record text. Rely heavily on the EHR Triage Note text as your ground-truth baseline context.
4. STRICT FORMATTING: You MUST output ONLY a valid JSON object. Do not include markdown blocks.

JSON STRUCTURE:
{
  "urgency_level": "Low" | "Medium" | "High" | "Critical",
  "suggested_next_action": "String detailing clinical next steps",
  "doctor_technical_alert": "String for the cardiologist",
  "family_reassurance_message": "String for the patient's family"
}
"""

    # Optional floats may arrive as None from the API layer's NaN handling.
    t_wave_amp_text = f"{t_wave_amplitude:.4f}" if t_wave_amplitude is not None else "not measured"
    st_elev_text = f"{st_elevation:.4f}" if st_elevation is not None else "not measured"

    user_prompt = f"""
Patient Clinical Profile for Verification:
- Patient ID: {patient_id}
- Source ECG Lead: {ecg_lead}
- Model Prediction Classification: {prediction} (Confidence: {confidence * 100:.1f}%)
- Rhythm Context: {rr_category} (Variance proxy: {rr_variance_proxy:.4f})
- Mathematical Dominance: {dominant_ecg_region}

[SHAP ATTRIBUTION EVIDENCE]
Signed values indicate direction of contribution; importance indicates magnitude.
- QRS complex: signed {qrs_shap_signed:.4f}, importance {qrs_importance:.4f}, direction {qrs_direction}
- P-wave: signed {p_wave_shap_signed:.4f}, importance {p_wave_importance:.4f}, direction {p_wave_direction}
- T-wave: signed {t_wave_shap_signed:.4f}, importance {t_wave_importance:.4f}, direction {t_wave_direction}

[PHYSIOLOGICAL FEATURES]
- P-wave present: {p_wave_present}
- T-wave amplitude: {t_wave_amp_text}
- ST elevation: {st_elev_text} (flag: {st_flag})

[BIO-BERT VALIDATED EHR CONTEXT]
- Confirmed Presenting Symptoms / History text: "{ehr_triage_note}"

Generate the grounded JSON response.
"""

    response = ollama.chat(
        model='monotykamary/medichat-llama3:8b',
        messages=[
            {'role': 'system', 'content': system_instruction},
            {'role': 'user',   'content': user_prompt}
        ],
        format='json',
        options={
            'temperature': 0.1,
            'num_predict': 400,
        }
    )
    return response['message']['content']


# ---------------------------------------------------------------------------
# CSV reader - validates all 23 expected columns
# ---------------------------------------------------------------------------

def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Load model_results_for_llm.csv and validate all required columns exist.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = {
        "patient_id", "ecg_lead",
        "true_label", "prediction", "confidence", "correct", "risk_flag",
        "qrs_shap_signed", "p_wave_shap_signed", "t_wave_shap_signed",
        "qrs_importance", "p_wave_importance", "t_wave_importance",
        "dominant_ecg_region",
        "qrs_direction", "p_wave_direction", "t_wave_direction",
        "rr_variance_proxy", "rr_category",
        "p_wave_present", "t_wave_amplitude", "st_elevation", "st_flag",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV '{csv_path}' is missing columns: {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            f"Ensure you ran the latest export script from Zafar's notebook."
        )
    return df


# ---------------------------------------------------------------------------
# Batch runner - processes every row using all 23 columns
# ---------------------------------------------------------------------------

def run_batch(csv_path: str, split_label: str) -> list:
    df = load_csv(csv_path)
    results = []

    print(f"\n[{split_label.upper()}] Processing {len(df)} records from '{csv_path}'...")

    for i, row in df.iterrows():
        patient_id = str(row["patient_id"])
        prediction = str(row["prediction"])

        ehr_triage_note = str(row.get("ehr_triage_note", "No clinical record history provided."))

        print(f"  [{i+1}/{len(df)}] {patient_id} - {prediction} "
              f"[risk: {row['risk_flag']}]", end="", flush=True)

        raw_response = ""
        try:
            raw_response = generate_agent_response(
                patient_id         = patient_id,
                ecg_lead           = str(row["ecg_lead"]),
                true_label         = str(row["true_label"]),
                prediction         = prediction,
                confidence         = float(row["confidence"]),
                correct            = bool(row["correct"]),
                risk_flag          = str(row["risk_flag"]),
                qrs_shap_signed    = float(row["qrs_shap_signed"]),
                p_wave_shap_signed = float(row["p_wave_shap_signed"]),
                t_wave_shap_signed = float(row["t_wave_shap_signed"]),
                qrs_importance     = float(row["qrs_importance"]),
                p_wave_importance  = float(row["p_wave_importance"]),
                t_wave_importance  = float(row["t_wave_importance"]),
                dominant_ecg_region= str(row["dominant_ecg_region"]),
                qrs_direction      = str(row["qrs_direction"]),
                p_wave_direction   = str(row["p_wave_direction"]),
                t_wave_direction   = str(row["t_wave_direction"]),
                rr_variance_proxy  = float(row["rr_variance_proxy"]),
                rr_category        = str(row["rr_category"]),
                p_wave_present     = int(row["p_wave_present"]),
                t_wave_amplitude=float(row["t_wave_amplitude"]) if pd.notna(row["t_wave_amplitude"]) else None,
                st_elevation=float(row["st_elevation"]) if pd.notna(row["st_elevation"]) else None,
                st_flag            = str(row["st_flag"]),
                ehr_triage_note=ehr_triage_note,
            )

            # Strip any accidental markdown wrapping before JSON parse
            clean_response = raw_response.replace("```json", "").replace("```", "").strip()
            agent_output   = json.loads(clean_response)
            status         = "success"

        except json.JSONDecodeError:
            agent_output = {"raw_response": raw_response}
            status       = "parse_error"
        except Exception as e:
            agent_output = {"error": str(e)}
            status       = "error"

        print(f" -> {status}")

        results.append({
            # -- Identity & prediction --
            "split":               split_label,
            "patient_id":          patient_id,
            "ecg_lead":            str(row["ecg_lead"]),
            "true_label":          str(row["true_label"]),
            "prediction":          prediction,
            "confidence":          float(row["confidence"]),
            "correct":             bool(row["correct"]),
            "risk_flag":           str(row["risk_flag"]),
            # -- SHAP evidence --
            "qrs_shap_signed":     float(row["qrs_shap_signed"]),
            "p_wave_shap_signed":  float(row["p_wave_shap_signed"]),
            "t_wave_shap_signed":  float(row["t_wave_shap_signed"]),
            "qrs_importance":      float(row["qrs_importance"]),
            "p_wave_importance":   float(row["p_wave_importance"]),
            "t_wave_importance":   float(row["t_wave_importance"]),
            "dominant_ecg_region": str(row["dominant_ecg_region"]),
            "qrs_direction":       str(row["qrs_direction"]),
            "p_wave_direction":    str(row["p_wave_direction"]),
            "t_wave_direction":    str(row["t_wave_direction"]),
            # -- RR & clinical features --
            "rr_variance_proxy":   float(row["rr_variance_proxy"]),
            "rr_category":         str(row["rr_category"]),
            "p_wave_present":      int(row["p_wave_present"]),
            "t_wave_amplitude":    float(row["t_wave_amplitude"]) if pd.notna(row["t_wave_amplitude"]) else None,
            "st_elevation":        float(row["st_elevation"])     if pd.notna(row["st_elevation"])     else None,
            "st_flag":             str(row["st_flag"]),
            # -- Log the grounded clinical text context --
            "ehr_triage_note": ehr_triage_note,
            # -- Agent output --
            "generated_at":        datetime.utcnow().isoformat() + "Z",
            "status":              status,
            "agent_output":        agent_output,
        })

    return results


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_json(results: list, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON saved -> {path} ({len(results)} records)")


def save_csv_flat(results: list, path: str) -> None:
    """Flatten agent_output fields into columns for dashboard consumption."""
    rows = []
    for r in results:
        rows.append({
            "split":                     r["split"],
            "patient_id":                r["patient_id"],
            "ecg_lead":                  r["ecg_lead"],
            "true_label":                r["true_label"],
            "prediction":                r["prediction"],
            "confidence":                r["confidence"],
            "correct":                   r["correct"],
            "risk_flag":                 r["risk_flag"],
            "qrs_shap_signed":           r["qrs_shap_signed"],
            "p_wave_shap_signed":        r["p_wave_shap_signed"],
            "t_wave_shap_signed":        r["t_wave_shap_signed"],
            "qrs_importance":            r["qrs_importance"],
            "p_wave_importance":         r["p_wave_importance"],
            "t_wave_importance":         r["t_wave_importance"],
            "dominant_ecg_region":       r["dominant_ecg_region"],
            "qrs_direction":             r["qrs_direction"],
            "p_wave_direction":          r["p_wave_direction"],
            "t_wave_direction":          r["t_wave_direction"],
            "rr_variance_proxy":         r["rr_variance_proxy"],
            "rr_category":               r["rr_category"],
            "p_wave_present":            r["p_wave_present"],
            "t_wave_amplitude":          r["t_wave_amplitude"],
            "st_elevation":              r["st_elevation"],
            "st_flag":                   r["st_flag"],
            "generated_at":              r["generated_at"],
            "status":                    r["status"],
            # LLM outputs
            "urgency_level":             r["agent_output"].get("urgency_level", ""),
            "suggested_next_action":     r["agent_output"].get("suggested_next_action", ""),
            "doctor_technical_alert":    r["agent_output"].get("doctor_technical_alert", ""),
            "family_reassurance_message":r["agent_output"].get("family_reassurance_message", ""),
        })

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"CSV saved -> {path} ({len(rows)} records)")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEST_CSV = "data/final_dataset_with_ehr.csv"
    JSON_OUT = "data/agent_outputs.json"
    CSV_OUT  = "data/agent_outputs.csv"

    all_results = []

    if Path(TEST_CSV).exists():
        test_results = run_batch(TEST_CSV, split_label="test")
        all_results.extend(test_results)
    else:
        print(f"[ERROR] {TEST_CSV} not found.")

    if all_results:
        save_json(all_results, JSON_OUT)
        save_csv_flat(all_results, CSV_OUT)
        print("\n=== agent_outputs.csv ready for dashboard. ===")
    else:
        print("No data processed. Check file paths.")
