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

const rows = json.map((item) => ({
  patient_external_id: item.patient_id,
  sample_id: item.sample_id ?? null,
  true_label: item.true_label ?? null,
  prediction: item.prediction ?? null,
  confidence: item.confidence ?? null,
  correct: item.correct === 1 || item.correct === true,
  risk_flag: item.risk_flag ?? null,
  dominant_ecg_region:
    item.dominant_ecg_region ?? item.dominullt_ecg_region ?? null,
  ecg_lead: item.ecg_lead ?? null,
  ehr_triage_note: item.ehr_triage_note ?? null,
  urgency_level: item.urgency_level ?? null,
  suggested_next_action: item.suggested_next_action ?? null,
  doctor_technical_alert: item.doctor_technical_alert ?? null,
  family_reassurance_message: item.family_reassurance_message ?? null,
  extracted_medical_entities: item.extracted_medical_entities ?? {},
  raw_output: item,
  generated_at: item.generated_at ?? new Date().toISOString(),
}));

console.log(`Importing ${rows.length} records...`);

const { error } = await supabase.from("agent_outputs").insert(rows);

if (error) {
  console.error(error);
  process.exit(1);
}

console.log("Import completed successfully.");