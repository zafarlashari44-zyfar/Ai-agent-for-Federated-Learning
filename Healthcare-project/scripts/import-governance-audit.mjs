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

const filePath = join(process.cwd(), "scripts", "ecg_audit_log.json");
const raw = await readFile(filePath, "utf8");
const audit = JSON.parse(raw);

const rows = audit.records.map((item) => ({
  patient_external_id: item.patient_id,
  sample_id: item.sample_index ?? null,
  true_label: item.true_class_aami ?? null,
  prediction: item.prediction ?? null,
  confidence: item.confidence ?? null,
  calibrated_confidence: item.calibrated_confidence ?? null,
  uncertainty_level: item.uncertainty_level ?? null,
  normalized_entropy: item.normalized_entropy ?? null,
  risk_score: item.risk_score ?? null,
  correct: item.ground_truth_miss === false,
  risk_flag:
    item.risk_score >= 0.45
      ? "High"
      : item.risk_score >= 0.25
        ? "Medium"
        : "Low",
  dominant_ecg_region: item.dominant_ecg_region ?? null,
  urgency_level: item.urgency_level ?? null,
  suggested_next_action: item.suggested_next_action ?? null,
  human_review_required: item.human_review_required ?? false,
  review_reason: item.review_reason ?? null,
  governance_policy: item.governance_policy ?? null,
  governance_route: item.route ?? null,
  clinical_review_status: item.clinical_review_status ?? null,
  review_mode: item.review_mode ?? null,
  hitl_released_at: item.hitl_released_at ?? null,
  low_confidence_flag: item.low_confidence_flag ?? false,
  disagreement_flag: item.disagreement_flag ?? false,
  blind_class_risk: item.blind_class_risk ?? false,
  exclusion_statement: null,
  raw_output: item,
  generated_at: item.logged_at ?? audit.generated_at ?? new Date().toISOString(),
}));

console.log(`Prepared ${rows.length} governance records.`);

const { error } = await supabase.from("agent_outputs").insert(rows);

if (error) {
  console.error(error);
  process.exit(1);
}

console.log("Governance audit import completed successfully.");
