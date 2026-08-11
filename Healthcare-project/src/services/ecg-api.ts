import { adaptECGAnalysisResponse } from "@/adapters/analysis-adapter";

import type {
  ECGAnalysisResult,
  ECGUploadRequestMetadata,
} from "@/types/ecg-analysis";

export interface ECGUploadPayload {
  files: File[];
  metadata?: ECGUploadRequestMetadata;
  includeExplanations?: boolean;
  includeOverlay?: boolean;
}

export class ECGApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details?: unknown;

  constructor(
    message: string,
    options: {
      status: number;
      code: string;
      details?: unknown;
    },
  ) {
    super(message);
    this.name = "ECGApiError";
    this.status = options.status;
    this.code = options.code;
    this.details = options.details;
  }
}

function appendMetadata(
  formData: FormData,
  metadata?: ECGUploadRequestMetadata,
) {
  if (!metadata) {
    return;
  }

  if (metadata.recordId) {
    formData.append("record_id", metadata.recordId);
  }

  if (metadata.selectedLead) {
    formData.append("lead_name", metadata.selectedLead);
  }

  if (
    typeof metadata.samplingRateHz === "number" &&
    Number.isFinite(metadata.samplingRateHz)
  ) {
    formData.append(
      "sampling_rate_hz",
      String(metadata.samplingRateHz),
    );
  }
}

function readErrorMessage(body: unknown) {
  if (
    typeof body === "object" &&
    body !== null
  ) {
    const record = body as Record<string, unknown>;

    if (typeof record.detail === "string") {
      return record.detail;
    }

    if (
      typeof record.error === "object" &&
      record.error !== null
    ) {
      const error = record.error as Record<string, unknown>;

      if (typeof error.message === "string") {
        return error.message;
      }
    }

    if (typeof record.error === "string") {
      return record.error;
    }
  }

  return "The ECG analysis request failed.";
}

export async function analyseECG(
  payload: ECGUploadPayload,
  signal?: AbortSignal,
): Promise<ECGAnalysisResult> {
  if (payload.files.length === 0) {
    throw new ECGApiError(
      "At least one ECG file is required.",
      {
        status: 400,
        code: "missing_ecg_file",
      },
    );
  }

  const formData = new FormData();

  payload.files.forEach((file) => {
    formData.append("files", file);
  });

  appendMetadata(formData, payload.metadata);

  formData.append(
    "include_explanations",
    String(payload.includeExplanations ?? true),
  );

  formData.append(
    "include_overlay",
    String(payload.includeOverlay ?? true),
  );

  formData.append(
    "waveform_downsample_limit",
    "20000",
  );

  formData.append(
    "overlay_downsample_limit",
    "20000",
  );

  const response = await fetch("/api/ecg/analyse", {
    method: "POST",
    body: formData,
    signal,
    cache: "no-store",
  });

  let body: unknown;

  try {
    body = await response.json();
  } catch {
    throw new ECGApiError(
      "The ECG analysis service returned invalid JSON.",
      {
        status: response.status,
        code: "invalid_json_response",
      },
    );
  }

  if (!response.ok) {
    throw new ECGApiError(
      readErrorMessage(body),
      {
        status: response.status,
        code: "analysis_request_failed",
        details: body,
      },
    );
  }

  return adaptECGAnalysisResponse(body);
}
