# Integration Guide — n8n Orchestration ↔ Federated Learning Pipeline

Connects the n8n clinical orchestration layer to the `Ai-agent-for-Federated-Learning`
repository without modifying a single file in that repository.

**Design principle:** their pipeline *ends* by writing artefacts to `outputs/`.
Yours *begins* by reading them. No code changes on their side, no merge conflicts,
and a clean contribution boundary for marking.

---

## Files in this bundle

| File | Goes where | Purpose |
|---|---|---|
| `build_cohort.py` | FL repo root | Joins their outputs into one cohort file |
| `app.py` | Your FastAPI folder | Widened payload; replaces the 5-field version |
| `ECG_Orchestration_v3_Integrated.json` | Import into n8n | The orchestration workflow |

---

## Step 1 — Place the bridge script

Copy `build_cohort.py` into the root of the FL repo, alongside `main_agent.py`.

```
Ai-agent-for-Federated-Learning/
├── main_agent.py
├── build_cohort.py        <-- here
├── fl_ecg_orchestrator/
└── outputs/
```

## Step 2 — Run their pipeline, then the bridge

```bash
cd Ai-agent-for-Federated-Learning
python main_agent.py --device cpu
python build_cohort.py --root .
```

Expected output:

```
Wrote outputs/orchestration/n8n_cohort.json
  records:   250
  unmatched: 0
  WARNING - model has ~zero recall on class(es): S, F
  67/250 records have a true class the model cannot detect.
```

If their outputs already exist you can skip `main_agent.py` and run the bridge alone.

## Step 3 — Replace your FastAPI app

Swap in the provided `app.py`. Keep `clinical_agent.py` exactly as it is —
nothing in it changes.

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Verify at `http://127.0.0.1:8000/docs`. The `PatientPayload` schema should now
show ~25 fields rather than 5.

## Step 4 — Import the workflow

n8n → **⋯ → Import from File** → `ECG_Orchestration_v3_Integrated.json`

## Step 5 — Set two file paths

Only two nodes need editing.

**"Read FL Cohort File"** → set *File(s) Selector* to the absolute path of
`outputs/orchestration/n8n_cohort.json`.

**"Write Audit File"** → set *File Name* to wherever the audit log should land.

Path format depends on how n8n runs:

| Setup | Cohort path |
|---|---|
| n8n native (npm/desktop), Windows | `C:\Users\you\...\outputs\orchestration\n8n_cohort.json` |
| n8n native, Mac/Linux | `/home/you/.../outputs/orchestration/n8n_cohort.json` |
| n8n in Docker | Container path — see below |

### If n8n runs in Docker

Two things change. Mount the repo:

```bash
docker run -it --rm \
  -p 5678:5678 \
  -v ~/.n8n:/home/node/.n8n \
  -v /absolute/path/to/Ai-agent-for-Federated-Learning:/data/repo \
  docker.n8n.io/n8nio/n8n
```

Then use `/data/repo/outputs/orchestration/n8n_cohort.json` as the cohort path,
and change the HTTP node URL from `127.0.0.1` to `host.docker.internal`
(on Linux add `--add-host=host.docker.internal:host-gateway`).

## Step 6 — Run it

Click **Execute workflow**. The Limit node caps the run at 10 patients so it
finishes in minutes rather than hours. Raise it for the full 250-record run that
produces your results table.

---

## What the integration actually does

```
FL PIPELINE (Zafar / Denis)                    ORCHESTRATION (you)
─────────────────────────                      ───────────────────
run_agentic_pipeline.py
  ├─ ShapAgent          → prediction_explanations.csv  ─┐
  ├─ UncertaintyAgent   → uncertainty_predictions.csv  ─┤
  ├─ CalibrationAgent   → calibration_metrics.json     ─┼→ build_cohort.py
  └─ EvaluationAgent    → per_class_metrics.csv        ─┘        │
                                                                 ▼
                                                        n8n_cohort.json
                                                                 │
                                                                 ▼
                                                    n8n: read → govern → gate
                                                         → audit → persist
```

Join key: `prediction_explanations.global_test_index == uncertainty_predictions.sample_index`
(250/250 matched).

---

## Governance thresholds and where each one comes from

Every gate traces to a measured artefact. None are guessed.

| Gate | Threshold | Source |
|---|---|---|
| MC-Dropout uncertainty | `uncertainty_level == High` | `uncertainty_predictions.csv` |
| Calibration-corrected confidence | `< 0.70` after ECE adjustment | `calibration_metrics.json` |
| Per-class reliability | `class_recall < 0.50` | `per_class_metrics.csv` |
| Composite risk score | `>= 0.45` | weighted, bounded [0,1] |
| Blind-class exposure | non-Low uncertainty + blind classes exist | `per_class_metrics.csv` |
| Agent urgency | High or Critical | LLM output |
| Structural verification | `status != success` | FastAPI |

**Why High uncertainty escalates:** their own `uncertainty_metrics.json` records
accuracy of **96.8% at Low uncertainty and 66.4% at High**. A 30-point collapse
is the empirical basis for the gate — not an assumption.

**Why confidence is corrected before use:** `confidence_gap` is −0.043 and ECE is
0.068. The model is measurably over-confident, so raw softmax is adjusted before
comparison against the floor.

---

## Three things to raise with your supervisor

**1. The model is blind to two classes.**
`per_class_metrics.csv` reports precision 0.0, recall 0.0, F1 0.0 for class **S**
(401 samples) and class **F**. In the 250-record cohort, **67 records have a true
class the model cannot detect**. This is the strongest argument in the dissertation
for why the governance layer exists — but it needs to be stated openly rather than
discovered by an examiner.

**2. There is no EHR text in the FL repo.**
`ehr_triage_note` is stubbed as absent. The clinical agent was designed around
grounded EHR context (rule 3 of its system prompt depends on it), so the family
and clinician messages are weaker than they should be. Either source the text or
document the limitation.

**3. SHAP region boundaries are inferred.**
`build_cohort.py` maps SHAP feature indices to P/QRS/T using constants from the
morphology extractor (`pre_r_window_ms=120`, `post_r_window_ms=160`, R-peak at
index 108 of a 216-sample window). **Confirm with Denis** before citing
SHAP-by-region in the methodology. The constants are at the top of the script.

---

## Scope note

`fl_ecg_orchestrator/config/build_config.py` already implements non-IID client
partitioning via simulated annealing, and `config.yaml` carries a full
`partition_audit` — 5 clients, per-client class counts, quality score 88.02.
That is roadmap steps J1 and J2, and Zafar has done it. Do not rebuild it and
do not claim it.

Your defensible contribution is the orchestration and governance layer. Worth
noting: the FL repo contains **no HTTP API of any kind** — it is entirely CLI
scripts driven by `main_agent.py`. Your FastAPI service is the only web interface
in the project, and the n8n workflow is the only inference-time governance layer.

Note also that the repo has its own `PlannerAgent` orchestrator. It is a
*training-time, batch* orchestrator that sequences SHAP/evaluation/calibration
agents. Yours is *inference-time, per-patient* with human gating. Different
layers — say so explicitly, or a marker may read it as duplicated effort.
