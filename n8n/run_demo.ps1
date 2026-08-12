<#
    run_demo.ps1 — one command, whole system, in demo order.

    USAGE
        cd "C:\Users\Jahid Shamim\Ai-agent-for-Federated-Learning\n8n"
        .\run_demo.ps1

    PREREQUISITES (all three must already be running)
        1. Ollama            -> check with:  ollama list
        2. uvicorn on 8001   -> uvicorn app:app --host 0.0.0.0 --port 8001
        3. n8n on 5678       -> n8n start      (only needed for the batch run)

    This script exercises the interactive layer only. It does not run the n8n
    batch workflow — do that from the n8n canvas so your supervisor sees the
    graph execute.
#>

$ErrorActionPreference = "Stop"
$base = "http://127.0.0.1:8001"

function Section($title) {
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor DarkCyan
    Write-Host "  $title" -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor DarkCyan
}

function Ask($persona, $question, $patientId) {
    $payload = @{ persona = $persona; question = $question }
    if ($patientId) { $payload.patient_id = $patientId }
    $body = $payload | ConvertTo-Json
    try {
        $r = Invoke-RestMethod -Uri "$base/api/ask" -Method Post -Body $body -ContentType "application/json"
        Write-Host ""
        Write-Host "Q: $question" -ForegroundColor Yellow
        if ($patientId) { Write-Host "   (scoped to $patientId)" -ForegroundColor DarkGray }
        Write-Host ""
        Write-Host $r.answer
    } catch {
        Write-Host "FAILED: $_" -ForegroundColor Red
    }
}

# ---------------------------------------------------------------- preflight
Section "PREFLIGHT"
try {
    $h = Invoke-RestMethod -Uri "$base/" -TimeoutSec 5
    Write-Host "  API up      : $($h.service) v$($h.version)" -ForegroundColor Green
} catch {
    Write-Host "  API NOT RUNNING on $base" -ForegroundColor Red
    Write-Host "  Start it with: uvicorn app:app --host 0.0.0.0 --port 8001" -ForegroundColor Red
    exit 1
}

$cohort = Invoke-RestMethod -Uri "$base/api/cohort/summary"
Write-Host "  Cohort      : $($cohort.record_count) records" -ForegroundColor Green
Write-Host "  Class split : $($cohort.true_class_distribution | ConvertTo-Json -Compress)" -ForegroundColor Green

$personas = Invoke-RestMethod -Uri "$base/api/personas"
Write-Host "  Personas    : $($personas.personas.id -join ', ')" -ForegroundColor Green

# ------------------------------------------------------------ 1. integration
Section "1. INTEGRATION WITH THE TEAM'S PIPELINE"
Write-Host @"
  build_cohort.py joins four artefacts the FL pipeline produces:
    - uncertainty_predictions.csv    (MC-Dropout entropy)
    - prediction_explanations.csv    (SHAP by ECG region)
    - calibration_metrics.json       (ECE)
    - per_class_metrics.csv          (per-class recall)
  Join key: global_test_index == sample_index.  250/250 matched, 0 unmatched.
  Every downstream decision carries the FL checkpoint name and round number.
"@
Write-Host "  Provenance from the live cohort file:" -ForegroundColor DarkGray
$cohort.provenance | ConvertTo-Json -Depth 3

# ------------------------------------------------------- 2. technical persona
Section "2. TECHNICAL PERSONA — architecture and methodology"
Ask "technical" "Why a 1D-CNN rather than an LSTM for this task?" $null
Ask "technical" "What are the measured weaknesses of this model?" $null

# ------------------------------------------------------- 3. clinician persona
Section "3. CLINICIAN PERSONA — per-patient evidence briefing"
$flagship = "PT-16878"
try {
    $p = Invoke-RestMethod -Uri "$base/api/patient/$flagship"
    Write-Host "  Record under review:" -ForegroundColor DarkGray
    Write-Host "    true class      : $($p.patient.true_class_aami)"
    Write-Host "    model predicted : $($p.patient.prediction)"
    Write-Host "    confidence      : $($p.patient.confidence)"
    Write-Host "    peers           : $($p.cohort_peers.peer_count) share this prediction, empirical accuracy $($p.cohort_peers.peer_empirical_accuracy)"
    Ask "clinician" "What did the model see here, and how much should I trust it?" $flagship
    Ask "clinician" "Are there other records like this one?" $flagship
} catch {
    Write-Host "  $flagship not in this cohort build - skipping." -ForegroundColor DarkYellow
}

# --------------------------------------------------------- 4. patient persona
Section "4. PATIENT PERSONA — plain language"
Ask "patient" "What does my result mean?" $flagship

# ------------------------------------------------------------- 5. monitoring
Section "5. SIMULATED PROFILES UNDER CONTINUOUS MONITORING"
$profiles = Invoke-RestMethod -Uri "$base/api/profiles"
Write-Host "  $($profiles.disclosure)" -ForegroundColor DarkYellow
Write-Host ""
Write-Host ("  {0,-9} {1,-22} {2,6} {3,6} {4,7}  {5}" -f "ID","CONTEXT","BEATS","ABNRM","UNDET","STATUS")
foreach ($p in $profiles.profiles) {
    $s = $p.summary; $m = $p.monitoring
    $colour = if ($m.status -eq "review_recommended") { "Red" } else { "Gray" }
    Write-Host ("  {0,-9} {1,-22} {2,6} {3,6} {4,7}  {5}" -f `
        $p.profile_id, $p.context.Substring(0, [Math]::Min(22, $p.context.Length)), `
        $s.beat_count, $s.abnormal_predictions, $s.beats_of_undetectable_class, $m.status) -ForegroundColor $colour
}

$flagged = $profiles.profiles | Where-Object { $_.monitoring.status -eq "review_recommended" } | Select-Object -First 1
if ($flagged) {
    Write-Host ""
    Write-Host "  Why $($flagged.profile_id) triggered:" -ForegroundColor Yellow
    foreach ($r in $flagged.monitoring.reasons) { Write-Host "    - $r" }
    Write-Host ""
    Write-Host "  $($flagged.monitoring.interpretation)" -ForegroundColor DarkYellow
}

# ----------------------------------------------------------------- 6. limits
Section "6. STATED LIMITATIONS"
Write-Host @"
  - Classes S and F have recall 0.0. The model cannot emit them. Class S has
    ROC-AUC 0.245, below chance: its ranking signal is inverted, not absent.
  - Class S fails CONFIDENTLY. Auto-cleared true-S beats carried mean calibrated
    confidence 0.948. Uncertainty-based gates cannot see them.
  - The governance layer caught 50 of 94 misclassifications (53.2%). 41 of the
    44 misses were true class S.
  - False alarm rate on correct predictions: 62.4%.
  - Patient identities and monitoring timestamps in section 5 are simulated.
  - The system classifies beats. It does not diagnose, and does not forecast
    cardiac events.
"@ -ForegroundColor DarkYellow

Write-Host ""
Write-Host "Demo complete." -ForegroundColor Green
Write-Host "For the batch orchestration run, execute the workflow at http://localhost:5678" -ForegroundColor Green
Write-Host ""
