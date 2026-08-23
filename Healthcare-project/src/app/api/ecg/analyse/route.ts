import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

const DEFAULT_ANALYSIS_ENDPOINT =
  "http://127.0.0.1:8000/api/v1/analyse";

const DEFAULT_N8N_URL =
  "http://localhost:5678";

const DEFAULT_N8N_WEBHOOK_PATH =
  "ecg-analysis";

function getAnalysisEndpoint() {
  const configuredEndpoint =
    process.env.ECG_ANALYSIS_API_URL?.trim();

  return configuredEndpoint || DEFAULT_ANALYSIS_ENDPOINT;
}

function getN8nEndpoint() {
  const baseUrl = (
    process.env.N8N_URL || DEFAULT_N8N_URL
  ).replace(/\/+$/, "");

  const webhookPath =
    process.env.N8N_WEBHOOK_PATH?.trim() ||
    DEFAULT_N8N_WEBHOOK_PATH;

  return `${baseUrl}/webhook/${webhookPath}`;
}

function createErrorResponse(
  status: number,
  error: string,
  message: string,
  details?: unknown,
) {
  return NextResponse.json(
    {
      success: false,
      error: {
        error,
        message,
        details,
      },
    },
    { status },
  );
}

function isRecord(
  value: unknown,
): value is Record<string, unknown> {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value)
  );
}

function asNumber(
  value: unknown,
  fallback = 0,
) {
  return typeof value === "number" &&
    Number.isFinite(value)
    ? value
    : fallback;
}

function asString(
  value: unknown,
  fallback = "",
) {
  return typeof value === "string"
    ? value
    : fallback;
}

function normalizedEntropy(
  probabilities: unknown,
) {
  if (!Array.isArray(probabilities)) {
    return 0;
  }

  const values = probabilities.filter(
    (value): value is number =>
      typeof value === "number" &&
      Number.isFinite(value) &&
      value >= 0,
  );

  if (values.length <= 1) {
    return 0;
  }

  const entropy = values.reduce(
    (total, probability) => {
      if (probability <= 0) {
        return total;
      }

      return (
        total -
        probability * Math.log(probability)
      );
    },
    0,
  );

  return entropy / Math.log(values.length);
}

function buildN8nPayload(
  payload: unknown,
) {
  if (!isRecord(payload)) {
    return null;
  }

  let analysis = payload;

  for (const key of [
    "result",
    "analysis",
    "analysis_result",
    "data",
  ]) {
    if (isRecord(payload[key])) {
      const candidate = payload[key];

      if (
        isRecord(candidate.prediction) &&
        isRecord(candidate.signal)
      ) {
        analysis = candidate;
        break;
      }
    }
  }

  const prediction = isRecord(
    analysis.prediction,
  )
    ? analysis.prediction
    : {};

  const signal = isRecord(
    analysis.signal,
  )
    ? analysis.signal
    : {};

  const recordingSummary = isRecord(
    analysis.recording_summary,
  )
    ? analysis.recording_summary
    : {};

  const confidence = asNumber(
    prediction.confidence,
    0,
  );

  const predictedLabel = asString(
    prediction.predicted_label,
    "Unknown",
  );

  const predictedClass =
    prediction.predicted_class;

  const classCodes = [
    "N",
    "S",
    "V",
    "F",
    "Q",
  ];

  const predictedAami =
    typeof predictedClass === "number"
      ? classCodes[predictedClass] ?? "Q"
      : "Q";

  const probabilities =
    prediction.probabilities;

  let uncertaintyLevel = "Low";

  if (confidence < 0.6) {
    uncertaintyLevel = "High";
  } else if (confidence < 0.8) {
    uncertaintyLevel = "Moderate";
  }

  const lead =
    asString(signal.lead_name) ||
    (
      Array.isArray(signal.lead_names) &&
      typeof signal.lead_names[0] ===
        "string"
        ? signal.lead_names[0]
        : "Unknown"
    );

  const recordId =
    asString(signal.record_id) ||
    "batch";

  return {
    patient_id: recordId,

    prediction:
      predictedLabel,

    confidence,

    uncertainty_level:
      uncertaintyLevel,

    normalized_entropy:
      normalizedEntropy(
        probabilities,
      ),

    qrs_shap_signed: 0,

    p_wave_shap_signed: 0,

    t_wave_shap_signed: 0,

    dominant_ecg_region:
      "QRS",

    class_recall: 0,

    ecg_lead:
      lead,

    true_class_aami:
      predictedAami,

    correct: false,

    blind_classes: [],

    total_beats:
      asNumber(
        recordingSummary.total_valid_beats,
        0,
      ),

    abnormal_beat_count:
      asNumber(
        recordingSummary.abnormal_beat_count,
        0,
      ),
  };
}

async function triggerEcgOrchestration(
  analysisPayload: unknown,
) {
  const secret =
    process.env.N8N_WEBHOOK_SECRET ||
    process.env.N8N_API_KEY ||
    "";

  if (!secret) {
    console.error(
      "ECG n8n orchestration skipped because N8N_WEBHOOK_SECRET is not configured.",
    );

    return;
  }

  const n8nPayload =
    buildN8nPayload(
      analysisPayload,
    );

  if (!n8nPayload) {
    console.error(
      "ECG n8n orchestration skipped because the analysis payload could not be mapped.",
    );

    return;
  }

  try {
    const response = await fetch(
      getN8nEndpoint(),
      {
        method: "POST",

        headers: {
          "Content-Type":
            "application/json",

          Authorization:
            `Bearer ${secret}`,
        },

        body: JSON.stringify(
          n8nPayload,
        ),

        cache: "no-store",

        signal:
          AbortSignal.timeout(
            15000,
          ),
      },
    );

    if (!response.ok) {
      const message =
        await response.text();

      console.error(
        `ECG n8n orchestration failed with status ${response.status}: ${message}`,
      );

      return;
    }

    console.log(
      "ECG n8n orchestration completed successfully.",
    );
  } catch (error) {
    console.error(
      "Unable to trigger ECG n8n orchestration:",
      error,
    );
  }
}

export async function POST(
  request: Request,
) {
  let incomingFormData: FormData;

  try {
    incomingFormData =
      await request.formData();
  } catch {
    return createErrorResponse(
      400,
      "invalid_form_data",
      "The ECG upload request could not be read.",
    );
  }

  const uploadedFiles =
    incomingFormData
      .getAll("files")
      .filter(
        (
          value,
        ): value is File =>
          value instanceof File,
      );

  const fallbackFile =
    incomingFormData.get(
      "file",
    );

  if (
    uploadedFiles.length === 0 &&
    !(fallbackFile instanceof File)
  ) {
    return createErrorResponse(
      400,
      "missing_ecg_file",
      "At least one ECG file is required.",
    );
  }

  const controller =
    new AbortController();

  const timeout = setTimeout(
    () => controller.abort(),
    5 * 60 * 1000,
  );

  try {
    const backendResponse =
      await fetch(
        getAnalysisEndpoint(),
        {
          method: "POST",

          body:
            incomingFormData,

          signal:
            controller.signal,

          cache:
            "no-store",
        },
      );

    const contentType =
      backendResponse.headers.get(
        "content-type",
      ) ?? "";

    if (
      contentType.includes(
        "application/json",
      )
    ) {
      const payload: unknown =
        await backendResponse.json();

      if (
        backendResponse.ok
      ) {
        await triggerEcgOrchestration(
          payload,
        );
      }

      return NextResponse.json(
        payload,
        {
          status:
            backendResponse.status,
        },
      );
    }

    const responseText =
      await backendResponse.text();

    if (!backendResponse.ok) {
      return createErrorResponse(
        backendResponse.status,
        "analysis_service_error",
        "The ECG analysis service returned an error.",
        responseText,
      );
    }

    return createErrorResponse(
      502,
      "invalid_analysis_response",
      "The ECG analysis service returned an unsupported response.",
      responseText,
    );
  } catch (error) {
    if (
      error instanceof Error &&
      error.name ===
        "AbortError"
    ) {
      return createErrorResponse(
        504,
        "analysis_timeout",
        "The ECG analysis exceeded the five minute timeout.",
      );
    }

    return createErrorResponse(
      502,
      "analysis_service_unavailable",
      "The FastAPI ECG analysis service could not be reached.",
      error instanceof Error
        ? error.message
        : String(error),
    );
  } finally {
    clearTimeout(timeout);
  }
}