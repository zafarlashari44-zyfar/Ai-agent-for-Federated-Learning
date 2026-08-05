import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

const DEFAULT_ANALYSIS_ENDPOINT =
  "http://127.0.0.1:8000/api/v1/analyse";

function getAnalysisEndpoint() {
  const configuredEndpoint =
    process.env.ECG_ANALYSIS_API_URL?.trim();

  return configuredEndpoint || DEFAULT_ANALYSIS_ENDPOINT;
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

export async function POST(request: Request) {
  let incomingFormData: FormData;

  try {
    incomingFormData = await request.formData();
  } catch {
    return createErrorResponse(
      400,
      "invalid_form_data",
      "The ECG upload request could not be read.",
    );
  }

  const uploadedFiles = incomingFormData
    .getAll("files")
    .filter((value): value is File => value instanceof File);

  const fallbackFile = incomingFormData.get("file");

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

  const controller = new AbortController();
  const timeout = setTimeout(
    () => controller.abort(),
    5 * 60 * 1000,
  );

  try {
    const backendResponse = await fetch(
      getAnalysisEndpoint(),
      {
        method: "POST",
        body: incomingFormData,
        signal: controller.signal,
        cache: "no-store",
      },
    );

    const contentType =
      backendResponse.headers.get("content-type") ?? "";

    if (contentType.includes("application/json")) {
      const payload = await backendResponse.json();

      return NextResponse.json(payload, {
        status: backendResponse.status,
      });
    }

    const responseText = await backendResponse.text();

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
      error.name === "AbortError"
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
      error instanceof Error ? error.message : String(error),
    );
  } finally {
    clearTimeout(timeout);
  }
}
