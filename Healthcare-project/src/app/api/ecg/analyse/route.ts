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
  )
    .trim()
    .replace(/\/+$/, "");

  const webhookPath = (
    process.env.N8N_WEBHOOK_PATH?.trim() ||
    DEFAULT_N8N_WEBHOOK_PATH
  ).replace(/^\/+/, "");

  return `${baseUrl}/webhook/${webhookPath}`;
}

function isUuid(value: string) {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(
    value,
  );
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
): number | undefined {
  return typeof value === "number" &&
    Number.isFinite(value)
    ? value
    : undefined;
}

function asString(
  value: unknown,
): string | undefined {
  return typeof value === "string" &&
    value.trim().length > 0
    ? value
    : undefined;
}

function mapAamiClass(
  predictedClass: unknown,
) {
  const classCodes = [
    "N",
    "S",
    "V",
    "F",
    "Q",
  ];

  return typeof predictedClass === "number"
    ? classCodes[predictedClass]
    : undefined;
}

function buildN8nPayload(
  payload: unknown,
  patientId: string,
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
    const candidate = payload[key];

    if (
      isRecord(candidate) &&
      isRecord(candidate.prediction)
    ) {
      analysis = candidate;
      break;
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

  const oodAssessment = isRecord(
    analysis.ood_assessment,
  )
    ? analysis.ood_assessment
    : {};

  const recordingSummary = isRecord(
    analysis.recording_summary,
  )
    ? analysis.recording_summary
    : {};

  return {
    patient_id: patientId,

    prediction:
      asString(
        prediction.predicted_label,
      ),

    predicted_class_aami:
      mapAamiClass(
        prediction.predicted_class,
      ),

    confidence:
      asNumber(
        prediction.confidence,
      ),

    normalized_entropy:
      asNumber(
        oodAssessment
          .normalized_prediction_entropy,
      ),

    ecg_lead:
      asString(
        signal.lead_name,
      ),

    pipeline_recommendation:
      asString(
        analysis.recommended_interpretation,
      ),

    input_accepted:
      analysis.input_accepted,

    model_prediction_produced:
      analysis.model_prediction_produced,

    signal_suitability:
      analysis.signal_suitability,

    ood_assessment:
      analysis.ood_assessment,

    analysis_scope:
      analysis.analysis_scope,

    reasoning:
      analysis.reasoning,

    clinical_report:
      analysis.clinical_report,

    analysis_warnings:
      analysis.analysis_warnings,

    total_beats:
      asNumber(
        recordingSummary.total_valid_beats,
      ),

    abnormal_beat_count:
      asNumber(
        recordingSummary.abnormal_beat_count,
      ),
  };
}

async function triggerEcgOrchestration(
  analysisPayload: unknown,
  patientId: string,
) {
  const secret =
    process.env.N8N_WEBHOOK_SECRET?.trim() ||
    process.env.N8N_API_KEY?.trim() ||
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
      patientId,
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

        body:
          JSON.stringify(
            n8nPayload,
          ),

        cache:
          "no-store",

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

  const patientIdValue =
    incomingFormData.get(
      "patient_id",
    );

  const patientId =
    typeof patientIdValue === "string"
      ? patientIdValue.trim()
      : "";

  if (!patientId) {
    return createErrorResponse(
      400,
      "missing_patient_id",
      "A real Supabase patient UUID is required for ECG analysis.",
    );
  }

  if (!isUuid(patientId)) {
    return createErrorResponse(
      400,
      "invalid_patient_id",
      "patient_id must be a valid Supabase UUID.",
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
          patientId,
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
