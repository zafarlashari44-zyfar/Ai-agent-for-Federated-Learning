"""
build_cohort.py

Bridges the FL pipeline (Zafar/Denis) into the n8n clinical orchestration
workflow (Jahid). Their side ends by dumping artefacts into outputs/, this
script picks up from there — joins everything into one record per sample
and writes a cohort file that n8n reads directly.

Inputs (all come from scripts/run_agentic_pipeline.py):
  outputs/explainability/shap/prediction_explanations.csv
  outputs/uncertainty/uncertainty_predictions.csv
  outputs/calibration/calibration_metrics.json
  outputs/evaluation/per_class_metrics.csv
  fl_ecg_orchestrator/config/config.yaml  (partition_audit)
  outputs/agentic_run/agent_run_manifest.json  (run provenance)

Output: outputs/orchestration/n8n_cohort.json

Join key is prediction_explanations.global_test_index against
uncertainty_predictions.sample_index.
"""

import csv
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone

# Region boundaries for a 216-sample R-peak-centred beat window @360Hz.
# Pulled from the reasoning_pipeline morphology extractor constants
# (pre_r_window_ms=120.0, post_r_window_ms=160.0).
#
# NOTE: confirm these with Denis before citing SHAP-by-region numbers
# in the dissertation - if his segmentation uses a different R-peak
# offset these boundaries move.
R_PEAK_INDEX = 108
QRS_START, QRS_END = 65, 166

AAMI_TO_CLINICAL = {
    "N": "Normal Sinus Rhythm",
    "S": "Supraventricular Ectopic",
    "V": "Ventricular Arrhythmia",
    "F": "Fusion Beat",
    "Q": "Unknown/Unclassifiable",
}


def region_of(index: int) -> str:
    if index < QRS_START:
        return "P_Wave"
    if index <= QRS_END:
        return "QRS"
    return "T_Wave"


def parse_shap_row(row):
    # top_feature_indices / top_shap_values are pipe-delimited strings,
    # e.g. "12|45|90" — split and bucket the signed values by region
    idxs = [int(x) for x in row["top_feature_indices"].split("|") if x != ""]
    vals = [float(x) for x in row["top_shap_values"].split("|") if x != ""]

    regions = {"P_Wave": 0.0, "QRS": 0.0, "T_Wave": 0.0}
    for i, v in zip(idxs, vals):
        regions[region_of(i)] += v

    dominant = max(regions, key=lambda k: abs(regions[k]))
    return regions, dominant, idxs, vals


def direction(value: float) -> str:
    if value > 0:
        return "increases prediction"
    if value < 0:
        return "decreases prediction"
    return "neutral"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Repository root")
    ap.add_argument("--out", default="outputs/orchestration/n8n_cohort.json")
    ap.add_argument("--limit", type=int, default=0, help="Cap records (0 = all)")
    args = ap.parse_args()

    root = Path(args.root).resolve()

    shap_path = root / "outputs/explainability/shap/prediction_explanations.csv"
    unc_path = root / "outputs/uncertainty/uncertainty_predictions.csv"
    cal_path = root / "outputs/calibration/calibration_metrics.json"
    percls_path = root / "outputs/evaluation/per_class_metrics.csv"
    manifest_path = root / "outputs/agentic_run/agent_run_manifest.json"

    # uncertainty, indexed by sample_index
    with open(unc_path, newline="", encoding="utf-8-sig") as f:
        unc = {r["sample_index"]: r for r in csv.DictReader(f)}

    # calibration applies to the whole cohort, not per-record
    with open(cal_path, encoding="utf-8-sig") as f:
        cal = json.load(f)

    # per-class reliability - which classes can we actually trust the model on
    class_reliability = {}
    with open(percls_path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            class_reliability[r["class"]] = {
                "precision": float(r["precision"]),
                "recall": float(r["recall"]),
                "f1_score": float(r["f1_score"]),
                "roc_auc": float(r["roc_auc"]),
                "support": int(float(r["support"])),
                # class the model basically never catches - hard escalation trigger
                "model_blind": float(r["recall"]) < 0.05,
            }

    # run provenance
    fl_run = {"checkpoint": None, "round": None}
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8-sig") as f:
            m = json.load(f)
        fl_run["checkpoint"] = Path(str(m.get("checkpoint", ""))).name or None
    if "round" in cal:
        fl_run["round"] = cal["round"]

    # --- join ---
    records = []
    unmatched = 0

    with open(shap_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if args.limit:
        rows = rows[: args.limit]

    for row in rows:
        gti = row["global_test_index"]
        u = unc.get(gti)
        if u is None:
            unmatched += 1
            continue

        regions, dominant, idxs, vals = parse_shap_row(row)

        pred_code = row["predicted_class"]
        true_code = row["true_class"]
        pred_clinical = AAMI_TO_CLINICAL.get(pred_code, pred_code)

        rel = class_reliability.get(pred_code, {})

        records.append({
            # identity / provenance
            "patient_id": f"PT-{int(gti):05d}",
            "sample_index": int(gti),
            "shap_sample_position": int(row["sample_position"]),
            "fl_checkpoint": fl_run["checkpoint"],
            "fl_round": fl_run["round"],
            "ecg_lead": "Lead II",

            # federated model prediction
            "true_class_aami": true_code,
            "predicted_class_aami": pred_code,
            "prediction": pred_clinical,
            "correct": row["correct"].strip().lower() == "true",
            "confidence": float(row["confidence"]),
            "confidence_level": row["confidence_level"],

            # MC dropout uncertainty (Zafar)
            "predictive_entropy": float(u["predictive_entropy"]),
            "normalized_entropy": float(u["normalized_entropy"]),
            "mutual_information": float(u["mutual_information"]),
            "predictive_variance": float(u["predictive_variance"]),
            "uncertainty_level": u["uncertainty_level"],
            "pipeline_recommendation": u["recommendation"],

            # SHAP attribution by ECG region (Denis)
            "p_wave_shap_signed": round(regions["P_Wave"], 6),
            "qrs_shap_signed": round(regions["QRS"], 6),
            "t_wave_shap_signed": round(regions["T_Wave"], 6),
            "p_wave_importance": round(abs(regions["P_Wave"]), 6),
            "qrs_importance": round(abs(regions["QRS"]), 6),
            "t_wave_importance": round(abs(regions["T_Wave"]), 6),
            "dominant_ecg_region": dominant,
            "p_wave_direction": direction(regions["P_Wave"]),
            "qrs_direction": direction(regions["QRS"]),
            "t_wave_direction": direction(regions["T_Wave"]),
            "top_feature_indices": idxs,

            # reliability of this specific predicted class
            "class_recall": rel.get("recall"),
            "class_precision": rel.get("precision"),
            "class_roc_auc": rel.get("roc_auc"),
            # almost always False - the model never predicts a class it can't
            # detect, so blind classes are invisible from the prediction side.
            # orchestrator should gate on uncertain predictions of emittable
            # classes instead - see blind_classes in cohort_metadata.
            "predicted_class_is_blind": rel.get("model_blind", False),

            # cohort-level calibration context
            "cohort_ece": cal.get("expected_calibration_error"),
            "cohort_confidence_gap": cal.get("confidence_gap"),

            # no EHR text exists in the FL repo, so this is just a placeholder
            "ehr_triage_note": "No clinical record text available for this record.",
            "ehr_source": "absent",
        })

    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "cohort_metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_pipeline": "fl_ecg_orchestrator + reasoning_pipeline",
            "fl_checkpoint": fl_run["checkpoint"],
            "fl_round": fl_run["round"],
            "record_count": len(records),
            "unmatched_rows": unmatched,
            "join_key": "prediction_explanations.global_test_index == uncertainty_predictions.sample_index",
            "calibration": cal,
            "class_reliability": class_reliability,
            "region_boundaries": {
                "r_peak_index": R_PEAK_INDEX,
                "qrs_start": QRS_START,
                "qrs_end": QRS_END,
                "note": "unconfirmed - verify against Denis's segmentation before citing",
            },
        },
        "records": records,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    print(f"Wrote {out_path}")
    print(f"  records:   {len(records)}")
    print(f"  unmatched: {unmatched}")
    blind = [c for c, r in class_reliability.items() if r["model_blind"]]
    if blind:
        print(f"  WARNING - model has ~zero recall on class(es): {', '.join(blind)}")
        exposed = sum(1 for r in records if r["true_class_aami"] in blind)
        print(f"  {exposed}/{len(records)} records have a true class the model cannot detect.")


if __name__ == "__main__":
    main()
