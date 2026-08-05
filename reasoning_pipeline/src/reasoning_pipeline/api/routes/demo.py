from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["demo"])


@router.get(
    "/demo",
    response_class=HTMLResponse,
    include_in_schema=False,
)
async def demo_page() -> HTMLResponse:
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >
    <title>ECG Reasoning Pipeline</title>

    <style>
        :root {
            color-scheme: dark;
            font-family: Inter, Arial, sans-serif;
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;
            background:
                radial-gradient(circle at top right, #183f4a, transparent 35%),
                linear-gradient(135deg, #061014, #0b1f25);
            color: #eefcff;
        }

        .container {
            width: min(1100px, calc(100% - 32px));
            margin: 0 auto;
            padding: 48px 0;
        }

        .header {
            margin-bottom: 28px;
        }

        h1 {
            margin: 0 0 10px;
            font-size: clamp(2rem, 6vw, 4rem);
            letter-spacing: -0.05em;
        }

        .subtitle {
            max-width: 760px;
            color: #a8c5cc;
            line-height: 1.6;
        }

        .grid {
            display: grid;
            grid-template-columns: minmax(300px, 0.8fr) minmax(0, 1.2fr);
            gap: 24px;
        }

        .card {
            border: 1px solid rgba(151, 227, 239, 0.17);
            border-radius: 20px;
            background: rgba(8, 29, 35, 0.82);
            box-shadow: 0 20px 80px rgba(0, 0, 0, 0.28);
            backdrop-filter: blur(14px);
            padding: 24px;
        }

        label {
            display: block;
            margin: 18px 0 7px;
            color: #c9edf2;
            font-size: 0.9rem;
        }

        input,
        button {
            width: 100%;
            border-radius: 10px;
            border: 1px solid rgba(151, 227, 239, 0.22);
            padding: 13px 14px;
            font: inherit;
        }

        input {
            background: #07181d;
            color: #ffffff;
        }

        button {
            margin-top: 22px;
            border: 0;
            background: #63d5e5;
            color: #041114;
            font-weight: 700;
            cursor: pointer;
        }

        button:disabled {
            opacity: 0.55;
            cursor: wait;
        }

        .status {
            min-height: 26px;
            margin-top: 16px;
            color: #9fdde6;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 18px;
        }

        .metric {
            border-radius: 14px;
            background: rgba(99, 213, 229, 0.08);
            padding: 16px;
        }

        .metric span {
            display: block;
            color: #8fb2b9;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .metric strong {
            display: block;
            margin-top: 7px;
            font-size: 1.1rem;
        }

        .section {
            margin-top: 18px;
            border-top: 1px solid rgba(151, 227, 239, 0.12);
            padding-top: 18px;
        }

        .section h3 {
            margin: 0 0 10px;
        }

        .section p {
            color: #c7dce0;
            line-height: 1.65;
            white-space: pre-wrap;
        }

        .hidden {
            display: none;
        }

        .error {
            color: #ffb7b7;
        }

        @media (max-width: 800px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>

<body>
    <main class="container">
        <header class="header">
            <h1>ECG Reasoning Pipeline</h1>
            <p class="subtitle">
                Upload a complete NumPy ECG recording. The system performs
                preprocessing, R-peak detection, beat segmentation,
                federated CNN inference, evidence construction,
                deterministic reasoning and report generation.
            </p>
        </header>

        <section class="grid">
            <div class="card">
                <h2>Analyse ECG</h2>

                <form id="analysis-form">
                    <label for="file">ECG recording (.npy)</label>
                    <input
                        id="file"
                        name="file"
                        type="file"
                        accept=".npy"
                        required
                    >

                    <label for="sampling-rate">Sampling rate (Hz)</label>
                    <input
                        id="sampling-rate"
                        name="sampling_rate_hz"
                        type="number"
                        value="360"
                        min="1"
                        step="0.1"
                        required
                    >

                    <label for="record-id">Record ID</label>
                    <input
                        id="record-id"
                        name="record_id"
                        type="text"
                        placeholder="Example: patient-001"
                    >

                    <label for="lead-name">Lead name</label>
                    <input
                        id="lead-name"
                        name="lead_name"
                        type="text"
                        placeholder="Example: MLII"
                    >

                    <button id="submit-button" type="submit">
                        Run ECG Analysis
                    </button>
                </form>

                <div id="status" class="status"></div>
            </div>

            <div id="result-card" class="card hidden">
                <h2>Analysis Result</h2>

                <div class="metric-grid">
                    <div class="metric">
                        <span>Prediction</span>
                        <strong id="prediction">—</strong>
                    </div>

                    <div class="metric">
                        <span>Model confidence</span>
                        <strong id="confidence">—</strong>
                    </div>

                    <div class="metric">
                        <span>Consistency</span>
                        <strong id="consistency">—</strong>
                    </div>

                    <div class="metric">
                        <span>Reasoning confidence</span>
                        <strong id="reasoning-confidence">—</strong>
                    </div>
                </div>

                <div class="section">
                    <h3>Clinical summary</h3>
                    <p id="clinical-summary"></p>
                </div>

                <div class="section">
                    <h3>Recommended action</h3>
                    <p id="recommended-action"></p>
                </div>

                <div class="section">
                    <h3>Reasoning conclusion</h3>
                    <p id="reasoning-conclusion"></p>
                </div>

                <div class="section">
                    <h3>Doctor narrative</h3>
                    <p id="doctor-report"></p>
                </div>

                <div class="section">
                    <h3>Patient / next-of-kin summary</h3>
                    <p id="next-of-kin-summary"></p>
                </div>
            </div>
        </section>
    </main>

    <script>
        const form = document.getElementById("analysis-form");
        const button = document.getElementById("submit-button");
        const statusElement = document.getElementById("status");
        const resultCard = document.getElementById("result-card");

        function percentage(value) {
            return `${(Number(value) * 100).toFixed(1)}%`;
        }

        form.addEventListener("submit", async (event) => {
            event.preventDefault();

            button.disabled = true;
            statusElement.className = "status";
            statusElement.textContent =
                "Running preprocessing, inference and reasoning...";
            resultCard.classList.add("hidden");

            const formData = new FormData(form);

            try {
                const response = await fetch("/api/v1/analyse", {
                    method: "POST",
                    body: formData,
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(
                        data.detail || "The ECG analysis failed."
                    );
                }

                document.getElementById("prediction").textContent =
                    data.prediction.predicted_label;

                document.getElementById("confidence").textContent =
                    percentage(data.prediction.confidence);

                document.getElementById("consistency").textContent =
                    data.reasoning.consistency_status;

                document.getElementById(
                    "reasoning-confidence"
                ).textContent = percentage(
                    data.reasoning.reasoning_confidence
                );

                document.getElementById(
                    "clinical-summary"
                ).textContent = data.clinical_report.summary;

                document.getElementById(
                    "recommended-action"
                ).textContent = data.clinical_report.recommended_action;

                document.getElementById(
                    "reasoning-conclusion"
                ).textContent = data.reasoning.conclusion;

                document.getElementById(
                    "doctor-report"
                ).textContent = data.narrative.doctor_report;

                document.getElementById(
                    "next-of-kin-summary"
                ).textContent = data.narrative.next_of_kin_summary;

                resultCard.classList.remove("hidden");
                statusElement.textContent = "Analysis completed successfully.";
            } catch (error) {
                statusElement.className = "status error";
                statusElement.textContent =
                    error instanceof Error
                        ? error.message
                        : "Unexpected analysis error.";
            } finally {
                button.disabled = false;
            }
        });
    </script>
</body>
</html>
"""
    )
