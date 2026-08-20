import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { createClient } from "@supabase/supabase-js";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
const serviceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!url || !serviceRoleKey) {
  console.error("Missing Supabase environment variables.");
  process.exit(1);
}

const supabase = createClient(url, serviceRoleKey);
const filePath = join(process.cwd(), "scripts", "agent_outputs.json");

const raw = await readFile(filePath, "utf8");
const json = JSON.parse(raw);

function resolvePayload(item) {
  return item.agent_output ?? item.agent_outputs ?? item.output ?? item;
}

const rows = json.map((item) => {
  const output = resolvePayload(item);

  return {
    patient_external_id:
      item.patient_id ??
      output.patient_id ??
      item.patient_external_id ??
      output.patient_external_id,

    sample_id: item.sample_id ?? output.sample_id ?? null,
    true_label: item.true_label ?? output.true_label ?? null,
    prediction: item.prediction ?? output.prediction ?? null,
    confidence: item.confidence ?? output.confidence ?? null,

    calibrated_confidence:
      item.calibrated_confidence ??
      output.calibrated_confidence ??
      null,

    uncertainty_level:
      item.uncertainty_level ??
      output.uncertainty_level ??
      null,

    normalized_entropy:
      item.normalized_entropy ??
      output.normalized_entropy ??
      null,

    correct:
      item.correct === 1 ||
      item.correct === true ||
      output.correct === 1 ||
      output.correct === true,

    risk_flag: item.risk_flag ?? output.risk_flag ?? null,
    risk_score: item.risk_score ?? output.risk_score ?? null,

    dominant_ecg_region:
      item.dominant_ecg_region ??
      output.dominant_ecg_region ??
      item.dominullt_ecg_region ??
      null,

    ecg_lead: item.ecg_lead ?? output.ecg_lead ?? null,
    ehr_triage_note:
      item.ehr_triage_note ??
      output.ehr_triage_note ??
      null,

    urgency_level:
      item.urgency_level ??
      output.urgency_level ??
      output.source_urgency_level ??
      null,

    suggested_next_action:
      item.suggested_next_action ??
      output.suggested_next_action ??
      null,

    doctor_technical_alert:
      item.doctor_technical_alert ??
      output.doctor_technical_alert ??
      null,

    family_reassurance_message:
      item.family_reassurance_message ??
      output.family_reassurance_message ??
      null,

    human_review_required:
      item.human_review_required ??
      output.human_review_required ??
      false,

    review_reason:
      item.review_reason ??
      output.review_reason ??
      null,

    governance_policy:
      item.governance_policy ??
      output.governance_policy ??
      null,

    governance_route:
      item.route ??
      output.route ??
      null,

    clinical_review_status:
      item.clinical_review_status ??
      output.clinical_review_status ??
      null,

    review_mode:
      item.review_mode ??
      output.review_mode ??
      null,

    hitl_released_at:
      item.hitl_released_at ??
      output.hitl_released_at ??
      null,

    low_confidence_flag:
      item.low_confidence_flag ??
      output.low_confidence_flag ??
      false,

    disagreement_flag:
      item.disagreement_flag ??
      output.disagreement_flag ??
      false,

    blind_class_risk:
      item.blind_class_risk ??
      output.blind_class_risk ??
      false,

    exclusion_statement:
      item.exclusion_statement ??
      output.exclusion_statement ??
      null,

    extracted_medical_entities:
      item.extracted_medical_entities ??
      output.extracted_medical_entities ??
      {},

    raw_output: item,

    generated_at:
      item.generated_at ??
      output.generated_at ??
      new Date().toISOString(),
  };
});

console.log(`Importing ${rows.length} records...`);

const { error } = await supabase.from("agent_outputs").insert(rows);

if (error) {
  console.error(error);
  process.exit(1);
}

console.log("Import completed successfully.");
