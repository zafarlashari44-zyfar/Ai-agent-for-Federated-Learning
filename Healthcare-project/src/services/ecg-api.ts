import type {
  ECGAnalysisApiResponse,
  ECGUploadRequestMetadata,
} from "@/types/ecg-analysis";

export interface ECGUploadPayload {
  files: File[];
  metadata?: ECGUploadRequestMetadata;
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
  if (!metadata) return;

  if (metadata.patientExternalId) {
    formData.append(
      "patient_external_id",
      metadata.patientExternalId,
    );
  }

  if (metadata.datasetName) {
    formData.append(
      "dataset_name",
      metadata.datasetName,
    );
  }

  if (metadata.recordId) {
    formData.append("record_id", metadata.recordId);
  }

  if (metadata.selectedLead) {
    formData.append(
      "lead_name",
      metadata.selectedLead,
    );
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

export async function analyseECG(
  payload: ECGUploadPayload,
  signal?: AbortSignal,
): Promise<unknown> {
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

  for (const file of payload.files) {
    formData.append("files", file);
  }

  appendMetadata(formData, payload.metadata);

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
    const apiResponse =
      body as Partial<ECGAnalysisApiResponse>;

    throw new ECGApiError(
      apiResponse.error?.message ??
        "The ECG analysis request failed.",
      {
        status: response.status,
        code:
          apiResponse.error?.error ??
          "analysis_request_failed",
        details: apiResponse.error?.details,
      },
    );
  }

  return body;
}
