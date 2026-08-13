# n8n ↔ Reasoning Pipeline integration

Refactors the orchestration layer to consume `POST /api/v1/analyse` as the
single source of clinical truth. No clinical analysis is reproduced in n8n.

Written against the real schema at
`reasoning_pipeline/src/reasoning_pipeline/api/schemas/analyse.py`
on `origin/api-service` (`b3f9286`). No field names are guessed.

---

## 1. Responsibility boundary as implemented

```
Raw ECG
   ↓
POST /api/v1/analyse          ← all clinical work happens here
   ↓
AnalysisResponse
   ↓
analyse_adapter.adapt()       ← maps to governance inputs, derives nothing clinical
   ↓
n8n                           ← risk scoring, escalation, HITL, notify, audit
```

---

## 2. Request contract

Read from `api/routes/analyse.py` on `origin/api-service` (`b3f9286`).

**`POST /api/v1/analyse` is `multipart/form-data`, not JSON.** The n8n HTTP
Request node needs Body Content Type = **Form-Data**, with `file` as a binary
property rather than a text field.

| Form field | Type | Required | Notes |
|---|---|---|---|
| `file` | binary | **yes** | One complete, **unsegmented** 1-D NumPy ECG recording. Beat segmentation happens server-side. Extension must be in the service's `supported_suffixes`. |
| `sampling_rate_hz` | float > 0 | **yes** | |
| `record_id` | str | no | Defaults to the uploaded filename stem |
| `lead_name` | str | no | |
| `include_explanations` | bool | no | Default `true` |
| `include_overlay` | bool | no | Default `true` |
| `overlay_start_sample` | int ≥ 0 | no | |
| `overlay_stop_sample` | int > 0 | no | |
| `overlay_downsample_limit` | int ≥ 1 | no | |

Upload limit: **25 MB**. Larger uploads return 413.

### Set both include flags to false for batch runs

`governance_request_form()` sends `include_explanations=false` and
`include_overlay=false` deliberately.

Governance reads `prediction`, `recording_summary`, `evidence`, `reasoning`,
`clinical_report` and `narrative`. It reads **nothing** from
`recording_explanation` or `recording_attribution_overlay` — those exist for
the frontend. Leaving the defaults on makes the pipeline compute per-beat
attribution maps and a full-resolution overlay for every recording, serialise
them, and send them, for data this layer discards. Across a queue that is a
large, avoidable cost on both sides.

Turn them on for a single recording a clinician is actively reviewing. Leave
them off for the batch.

### Error handling

All four documented error statuses mean the recording was **not analysed**.
`failsafe_record()` escalates each to human review rather than treating it as a
clear result — the same principle as the previous architecture's
agent-unreachable branch.

| Status | Meaning |
|---|---|
| 413 | Upload exceeds 25 MB |
| 415 | Unsupported ECG file format |
| 422 | Malformed request or unprocessable recording |
| 503 | Pipeline service unavailable |

---

## 3. Mapping table

| API field | n8n field | Purpose |
|---|---|---|
| `signal.record_id` | `record_id` | Audit key, replaces synthetic `patient_id` |
| `signal.duration_seconds` | `duration_seconds` | Context on audit record |
| `signal.lead_name` | `lead_name` | Context on audit record |
| `prediction.predicted_label` | `predicted_label` | Displayed; never re-derived |
| `prediction.confidence` | `prediction_confidence` | Confidence-floor gate |
| `prediction.probabilities` | → `recording_entropy` | **Derived.** Decisiveness gate |
| `prediction.checkpoint_hash` | `checkpoint_hash` | Provenance stamp on every audit record |
| `prediction.model_version` | `model_version` | Provenance |
| `prediction.preprocessing_version` | `preprocessing_version` | Provenance |
| `recording_summary.total_valid_beats` | `total_valid_beats` | Denominator for burden |
| `recording_summary.abnormal_beat_count` | `abnormal_beat_count` | Burden gate |
| `recording_summary.abnormal_beat_percentage` | → `abnormal_fraction` | **Derived** (÷100). Burden gate |
| `recording_summary.class_counts` | `class_counts` | Audit detail |
| `recording_summary.dominant_predicted_label` | `dominant_label` | Audit detail |
| `recording_summary.beat_results[].prediction.probabilities` | → `high_entropy_beat_count` | **Derived.** Per-beat decisiveness |
| `evidence.supporting[].reliability` | → `low_reliability_evidence_count` | **Derived.** Evidence-quality gate |
| `evidence.conflicting` | (count) | Escalation gate |
| `evidence.evidence_version` | `evidence_version` | Provenance |
| `reasoning.consistency_status` | `consistency_status` | **Primary escalation gate** |
| `reasoning.reasoning_confidence` | `reasoning_confidence` | Audit detail |
| `reasoning.conclusion` | `conclusion` | Displayed verbatim |
| `reasoning.rule_trace` | `rule_trace` | Audit — the deterministic trace |
| `clinical_report.summary` | `clinical_summary` | Displayed verbatim |
| `clinical_report.recommended_action` | `recommended_action` | Displayed verbatim; n8n does **not** generate this |
| `clinical_report.supporting_findings` | `supporting_findings` | Displayed verbatim |
| `clinical_report.conflicting_findings` | `conflicting_findings` | Displayed verbatim |
| `clinical_report.limitations` | `clinical_limitations` | Displayed verbatim |
| `narrative.doctor_report` | `doctor_report` | Clinician alert body |
| `narrative.next_of_kin_summary` | `next_of_kin_summary` | Family-facing notification |
| `narrative.fallback_used` | `narrative_fallback_used` | Escalation gate — template fallback means degraded output |
| `recording_explanation.warnings` | (count) | Escalation gate |
| `recording_attribution_overlay` | not consumed | Frontend transport; no governance use |

---

## 4. Duplicated logic removed

| Removed | Previously | Now |
|---|---|---|
| ECG interpretation | Local LLM call in `n8n/app.py` | `prediction`, `recording_summary` |
| Clinical reasoning | LLM prompt producing urgency | `reasoning` |
| Recommendation generation | LLM producing `suggested_next_action` | `clinical_report.recommended_action` |
| Clinical report generation | Assembled in the FastAPI bridge | `clinical_report` |
| Narrative generation | `clinical_agent.py` doctor/family text | `narrative` |
| XAI generation | SHAP-by-region read from CSV, with **inferred** region boundaries | `recording_explanation` |
| Evidence generation | Composed from cohort CSV columns | `evidence` |
| Cohort file join | `build_cohort.py` merging four artefacts | Not needed — API returns a complete result per recording |

`clinical_agent.py` and the LLM path in `app.py` are no longer on the runtime
path. The persona layer (`persona_service.py`) can stay: it answers questions
*about* results and does not produce them, but it should be repointed at the
API response rather than the cohort file.

---

## 5. Governance responsibilities retained

Owned by n8n, unchanged in principle:

- Human-in-the-loop routing and the review queue
- Escalation thresholds and the composite decision
- Notification workflows
- Audit logging with provenance and integrity checksum
- Retry and fail-safe handling (agent unreachable → escalate, never drop)
- Queue management and batching
- Operational metrics

---

## 6. Governance assumptions checked against the API

| Assumption | Status |
|---|---|
| **Blind classes S and F** | Now **verified at runtime**, not assumed. `verify_blind_classes()` tests the declared labels against emitted probability mass and flags the assumption as stale if either carries >1%. The old gate asserted this from an evaluation CSV and could not detect drift. |
| **MC-Dropout uncertainty band** | **Not available.** Replaced by normalised entropy over `prediction.probabilities` — computed from the deployed checkpoint rather than a separate offline artefact. |
| **ECE-corrected confidence** | **Not available, and deliberately not reconstructed.** The previous policy subtracted ECE 0.0684 from raw confidence on the basis that the model was overconfident. The measured `confidence_gap` was **−0.0435** — mean confidence *below* accuracy, i.e. under-confident. The correction was likely the wrong direction and a probable cause of the 62.4% false-alarm rate. Raw `prediction.confidence` is now used directly. |
| **Per-class recall floor** | **Not available.** Belongs to the evaluation pipeline. Would need the API to expose per-class reliability for the deployed checkpoint. |
| **SHAP region boundaries** | **No longer needed.** The old gate used P-wave/QRS/T-wave windows *inferred* from R-peak alignment and never confirmed. `recording_explanation` supplies real attribution with explicit sample indices. |
| **Ground-truth miss detection** | **Not available, correctly.** The API does inference on raw ECG and returns no labels. Retrospective accuracy is an evaluation activity, not runtime governance. Removed from the runtime path. |

---

## 7. API fields that would improve integration

Requests for the pipeline team, in priority order:

1. **Per-class reliability for the deployed checkpoint** — recall or a
   `model_blind` flag per class, ideally on `PredictionResponse`. This would
   let the blind-class gate cite the deployed model's own metrics instead of
   inferring from probability mass.
2. **Calibration metadata** — an ECE or reliability-curve reference tied to
   `checkpoint_hash`, with a stated direction. Without it, confidence cannot be
   corrected and the floor is applied to raw values.
3. **A stable severity or triage field on `ClinicalReport`** — currently
   `recommended_action` is free text, so governance cannot key on it without
   string matching, which is brittle.
4. **`suitability` and `ood_assessment` as first-class response fields** — the
   spec describes both as pipeline stages but neither appears in
   `AnalysisResponse`. If they exist, exposing them would give governance two
   strong upstream gates. **Worth confirming: were these intended to be in the
   response?**

---

## Status and blockers

### Resolved
- **Request contract** — read from the route file; multipart, fields documented above.
- **Calibration** — agreed with the pipeline team: no ECE correction in n8n. Raw `prediction.confidence` is used and n8n owns the operational thresholds.
- **v7 results framing** — agreed: the 250-patient run is the **previous offline orchestration evaluation**. The live `/analyse` integration is a different architecture and is not expected to reproduce those numbers until the integrated system is evaluated separately.

### Blocked on the pipeline team
`signal_suitability`, `ood_assessment` and `analysis_scope` are **not present**
on `origin/api-service` at `b3f9286`. Verified by grep against the schema on
that commit — zero matches. The pipeline team reports these exist locally; they
need pushing before the adapter can consume them.

Once pushed, they become the **first gates**, ahead of confidence and risk
routing, per the pipeline team's guidance: a recording that fails suitability
or is flagged out-of-distribution should never reach confidence-based routing,
because a confidence value on an unsuitable signal is not meaningful.

### Still to calibrate
`CONFIDENCE_FLOOR`, `ENTROPY_CEILING` and `ABNORMAL_BURDEN_CEILING` in
`analyse_adapter.py` are provisional policy defaults, not derived from a
calibration run against this API. They must be tuned against real responses
before any escalation rate from the integrated system is quoted.
