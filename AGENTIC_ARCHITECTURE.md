# Agentic Federated ECG Architecture

This module restores the original master-agent design without deleting or
rewriting the teammate's `agent1_scribe.py`.

## Agents

- `ScribeAgent`: adapter around the teammate's ECG/clinical-text scribe.
- `CheckpointAgent`: locates and validates the trained federated checkpoint.
- `ShapAgent`: delegates to the existing SHAP runner.
- `AdvancedEvaluationAgent`: delegates to the existing advanced evaluator.
- `CalibrationAgent`: delegates to the existing calibration runner.
- `UncertaintyAgent`: delegates to the existing Monte Carlo Dropout runner.
- `ReportAgent`: combines agent status and generated research JSON outputs.
- `PlannerAgent`: master orchestrator controlling order and failure policy.

## Standard run

```powershell
python .\main_agent.py --device cpu
```

## Faster test without repeating SHAP

```powershell
python .\main_agent.py --device cpu --skip-shap --mc-samples 10
```

## Run the teammate Scribe agent too

The Scribe requires its original dependencies and suitable raw MIT-BIH record
paths. Its source remains separate and credited to its original contributor.

```powershell
python .\main_agent.py `
  --device cpu `
  --run-scribe `
  --scribe-path .\agent1_scribe.py `
  --record-name path\to\mitdb\100
```

## Output

```text
outputs\agentic_run\agent_run_manifest.json
outputs\agentic_run\agent_run_summary.txt
```
