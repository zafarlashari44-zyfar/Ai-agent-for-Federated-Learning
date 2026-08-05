create table if not exists public.agent_outputs (
  id uuid primary key default extensions.gen_random_uuid(),
  patient_external_id text not null,
  sample_id integer,
  true_label text,
  prediction text,
  confidence numeric,
  correct boolean,
  risk_flag text,
  dominant_ecg_region text,
  ecg_lead text,
  ehr_triage_note text,
  urgency_level text,
  suggested_next_action text,
  doctor_technical_alert text,
  family_reassurance_message text,
  extracted_medical_entities jsonb not null default '{}'::jsonb,
  raw_output jsonb not null,
  generated_at timestamptz,
  created_at timestamptz not null default now()
);

create index if not exists agent_outputs_patient_external_idx
on public.agent_outputs (patient_external_id);

create index if not exists agent_outputs_risk_created_idx
on public.agent_outputs (risk_flag, created_at desc);