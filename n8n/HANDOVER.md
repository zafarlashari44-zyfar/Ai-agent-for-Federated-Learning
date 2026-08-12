# HANDOVER — Agentic Federated Learning, orchestration & governance layer

Everything below is built, tested and running. Read the **Demo day** section
first; the rest is reference.

---

## Files and where they go

| File | Destination | Purpose |
|---|---|---|
| `app.py` | `n8n\` | FastAPI service v4.0.0 — batch + interactive |
| `persona_service.py` | `n8n\` | Technical / clinician / patient personas |
| `profiles_service.py` | `n8n\` | Simulated profiles + continuous monitoring |
| `model_card.json` | `n8n\` | The only facts the technical persona may claim |
| `run_demo.ps1` | `n8n\` | One-command walkthrough of the whole system |
| `ECG_Orchestration_v7_READY.json` | import into n8n | Batch workflow |
| `build_cohort.py` | repo root | Joins the team's pipeline artefacts |

Copy them in one go:

```powershell
Copy-Item "$HOME\Downloads\app.py","$HOME\Downloads\persona_service.py",`
          "$HOME\Downloads\profiles_service.py","$HOME\Downloads\model_card.json",`
          "$HOME\Downloads\run_demo.ps1" `
          "C:\Users\Jahid Shamim\Ai-agent-for-Federated-Learning\n8n\" -Force
```

---

## Cold start — four steps

```powershell
# 1. Ollama (starts as a Windows service; just verify)
ollama list

# 2. Open the REPO folder in VS Code (not the n8n subfolder - the venv
#    lives at the repo root and only activates when that folder is open).
#    Terminal prompt must show (.venv).

# 3. Pane 1 - the API. Leave it running.
cd n8n
uvicorn app:app --host 0.0.0.0 --port 8001

# 4. Pane 2 (Ctrl+Shift+5) - n8n, only needed for the batch run.
n8n start
```

On startup pane 1 should print:

```
[API] Persona layer mounted: /api/ask, /api/patient/{id}, /api/cohort/summary, /api/personas
[API] Profiles layer mounted: /api/profiles, /api/profiles/{id}, /api/profiles/{id}/stream
```

If either says WARNING, the corresponding `.py` file is not in `n8n\`.

"Address already in use" from uvicorn or n8n means it is already running.
Not an error — skip that step.

---

## Demo day

Two things to show, in this order.

### A. The interactive layer — one command

```powershell
cd "C:\Users\Jahid Shamim\Ai-agent-for-Federated-Learning\n8n"
.\run_demo.ps1
```

Six sections, roughly four minutes: integration provenance, technical persona,
clinician persona on a real misclassified patient, patient persona, simulated
profiles under monitoring, stated limitations.

If PowerShell blocks the script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### B. The batch orchestration — on the n8n canvas

Open `http://localhost:5678`, open the v7 workflow, press **Execute workflow**.
Set **Limit (demo cap)** to 10 for a live demo (about one minute); 250 takes
roughly 18 minutes.

Let her watch the item counts move along the connections. That is the
orchestration story: 250 in, split, agent call, governance, review gate, audit.

---

## What answers each of her three questions

**"Does it integrate with their work?"**
Yes, at artefact level, and it is evidenced. `build_cohort.py` joins Zafar's
uncertainty output and Denis's SHAP output on `global_test_index ==
sample_index` — 250/250 matched, 0 unmatched. Every audit record carries the FL
checkpoint name and round. Governance thresholds derive from their measured
metrics, not from guesses.

Be precise about the boundary: this reads their **output files**, it does not
call their **code**. If she wants service-level integration, the HTTP node can
be repointed at the team's `/analyse` endpoint — one node change, but a team
decision about who owns the reasoning service.

**"Does it send notifications?"**
Node is built and wired into the escalation branch, shipped disabled so the
workflow runs without credentials. To enable: click **Notify Clinician
(Email)** → Credential → Create new → Gmail OAuth2 → set Send To → right-click
→ Enable. **Set Limit to 3 before testing**, or you will send yourself 54
emails.

**"Does it work with other datasets?"**
Architecturally yes — the workflow consumes a *contract* (the four artefact
files), not a dataset. Re-run the FL pipeline on different data, re-run
`build_cohort.py`, and the workflow needs no changes. This has **not yet been
demonstrated end to end**; say "designed for it, not yet shown" rather than
claiming it.

---

## Results from the full run — the numbers to cite

250 patients, 18m 26s, 227 successful agent calls, 23 transport failures.

| Metric | Value |
|---|---|
| Escalation rate | 62.4% |
| Model misclassifications | 94 of 227 (41.4%) |
| Misclassifications caught by governance | 50 (**53.2%**) |
| Misclassifications missed | 44 |
| False alarms on correct predictions | 83 of 133 (62.4%) |

### The finding worth leading with

**41 of the 44 misses are true class S.** The model's per-class recall for S is
0.0, and its ROC-AUC for S is **0.245 — below chance**, meaning the ranking
signal is inverted rather than merely absent.

Crucially, these are *confident* failures:

| | Auto-cleared true-S | Escalated records |
|---|---|---|
| Calibrated confidence | 0.948 | 0.589 |
| Normalised entropy | 0.039 | — |
| MC-Dropout uncertainty | Low, all 41 | — |

The argument this supports:

> Uncertainty-based governance catches unreliable predictions but is
> structurally blind to confident systematic failure. A federated model that has
> never learned class S does not hesitate — it emits high-confidence normal
> readings. MC-Dropout entropy, calibrated confidence and LLM disagreement all
> fail together because they measure the same underlying signal.

Class F behaves differently (86.7% caught) because those predictions *are*
uncertain — same zero recall, different failure mode. That contrast is the
interesting part.

---

## Open issues — raise these before she finds them

1. **Calibration direction may be wrong.** Governance computes
   `calibrated = raw_confidence − ECE(0.0684)` on the stated basis that the
   model is overconfident. But `confidence_gap = −0.0435` (mean confidence
   0.8847 vs accuracy 0.9282) means it is **under**-confident on aggregate. ECE
   is direction-agnostic. Subtracting is likely wrong and is probably inflating
   the 62.4% false-alarm rate. Documented in `model_card.json`; not silently
   changed, because it affects results you may already have drafted.

2. **Overall accuracy is misleading.** 92.8% on a cohort that is ~85% class N is
   near the majority-class baseline. Never quote it without that caveat. The
   personas are instructed to state the caveat automatically.

3. **HITL gate is a 5-second timed hold**, not a real sign-off. Switch the Wait
   node to Resume → On Webhook Call before calling it genuine
   human-in-the-loop in the write-up.

4. **Beat labels only.** Atrial fibrillation, bradycardia and ventricular
   tachycardia are *rhythm* annotations — a separate MIT-BIH label set your
   pipeline does not use. Your teammates appear to be building toward this:
   `reasoning_pipeline/tests/unit/scribe_v2/test_rhythm_feature_extractor.py`
   exists. **Ask them what is in it before promising rhythm analysis.**

5. **No cardiac-arrest prediction, and there cannot be.** A per-beat classifier
   cannot forecast a future event. The monitoring layer reports an
   *evidence-accumulation trigger* over a sliding window, labelled as such in
   every response. Do not let it be described as prediction.

6. **23 transport failures** persisted even with request batching. Local 8B
   inference under sustained load is not fully reliable. The fail-safe escalated
   all 23 rather than dropping them — correct behaviour, and worth saying so.

7. **SHAP region boundaries unconfirmed.** The cohort provenance itself says
   `"note": "unconfirmed - verify against Denis's segmentation before citing"`.
   Confirm with Denis before citing SHAP-by-region.

8. **EHR triage text is stubbed.** No clinical notes exist in the repo.

---

## Simulated profiles — what is real

Six profiles, 250 real cohort beats distributed deterministically between them.

**Real:** every beat, its true label, model prediction, confidence and
MC-Dropout uncertainty, straight from the FL pipeline. The per-class reliability
figures behind the flags.

**Simulated:** the people. Names, ages, clinical context, device, and the
monitoring timestamps. Beat-to-patient assignment is invented.

The API says this in a `disclosure` field on every response, and `simulated:
true` on every profile. Say it out loud in the demo too.

The monitoring trigger fires when a recent window is materially worse than the
**cohort's own base rate** — thresholds derive from the data, not from guessed
constants, the same principle as the governance layer. With fixed thresholds all
six profiles fired at once and the signal was useless.

---

## Still to do — in priority order if time allows

1. **Message Zafar and Denis** about the overlapping branches (`api-service`,
   `reasoning-engine`, `narrative-generator`). Two parallel FastAPI services
   doing similar work is the thing most likely to be flagged at marking. This is
   a conversation, not a code change, and it matters more than any remaining
   feature.
2. Resolve the calibration direction, or write it up as a known issue.
3. Demonstrate the second dataset end to end.
4. Enable and test the Gmail node with Limit = 3.
5. Switch the Wait node to webhook resume.
