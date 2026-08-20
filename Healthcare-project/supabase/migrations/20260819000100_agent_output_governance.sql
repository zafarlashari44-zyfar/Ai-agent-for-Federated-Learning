alter table public.agent_outputs
  add column if not exists calibrated_confidence numeric,
  add column if not exists uncertainty_level text,
  add column if not exists normalized_entropy numeric,
  add column if not exists risk_score numeric,
  add column if not exists human_review_required boolean not null default false,
  add column if not exists review_reason text,
  add column if not exists governance_policy text,
  add column if not exists governance_route text,
  add column if not exists clinical_review_status text,
  add column if not exists review_mode text,
  add column if not exists hitl_released_at timestamptz,
  add column if not exists low_confidence_flag boolean not null default false,
  add column if not exists disagreement_flag boolean not null default false,
  add column if not exists blind_class_risk boolean not null default false,
  add column if not exists exclusion_statement text;

create index if not exists agent_outputs_human_review_created_idx
on public.agent_outputs (human_review_required, created_at desc);
