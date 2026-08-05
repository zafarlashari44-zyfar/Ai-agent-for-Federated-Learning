import type {
  AamiClass,
  ECGAnalysisApiResponse,
  ECGAnalysisResult,
} from "@/types/ecg-analysis";

const AAMI_CLASSES = new Set<AamiClass>([
  "N",
  "S",
  "V",
  "F",
  "Q",
]);

function isRecord(
  value: unknown,
): value is Record<string, unknown> {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value)
  );
}

function isAamiClass(value: unknown): value is AamiClass {
  return (
    typeof value === "string" &&
    AAMI_CLASSES.has(value as AamiClass)
  );
}

function isAnalysisResult(
  value: unknown,
): value is ECGAnalysisResult {
  if (!isRecord(value)) return false;

  const metadata = value.metadata;
  const waveform = value.waveform;
  const prediction = value.prediction;
  const reasoning = value.reasoning;
  const uncertainty = value.uncertainty;

  if (
    typeof value.analysisId !== "string" ||
    typeof value.generatedAt !== "string" ||
    !isRecord(metadata) ||
    !isRecord(waveform) ||
    !isRecord(prediction) ||
    !isRecord(reasoning) ||
    !isRecord(uncertainty)
  ) {
    return false;
  }

  if (
    typeof metadata.samplingRateHz !== "number" ||
    typeof metadata.durationSeconds !== "number" ||
    !Array.isArray(waveform.points) ||
    !Array.isArray(value.rPeaks) ||
    !Array.isArray(value.beats) ||
    !Array.isArray(value.attributionRegions)
  ) {
    return false;
  }

  return (
    isAamiClass(prediction.classCode) &&
    typeof prediction.confidence === "number"
  );
}

export class ECGAnalysisAdapterError extends Error {
  readonly payload: unknown;

  constructor(message: string, payload: unknown) {
    super(message);
    this.name = "ECGAnalysisAdapterError";
    this.payload = payload;
  }
}

export function adaptECGAnalysisResponse(
  payload: unknown,
): ECGAnalysisResult {
  if (isAnalysisResult(payload)) {
    return payload;
  }

  if (isRecord(payload)) {
    const wrapped =
      payload as Partial<ECGAnalysisApiResponse>;

    if (wrapped.success === false) {
      throw new ECGAnalysisAdapterError(
        wrapped.error?.message ??
          "The ECG analysis service reported a failure.",
        payload,
      );
    }

    if (isAnalysisResult(wrapped.result)) {
      return wrapped.result;
    }

    const candidateKeys = [
      "analysis",
      "analysis_result",
      "result",
      "data",
    ];

    for (const key of candidateKeys) {
      const candidate = payload[key];

      if (isAnalysisResult(candidate)) {
        return candidate;
      }
    }
  }

  throw new ECGAnalysisAdapterError(
    "The ECG analysis response does not match the dashboard contract.",
    payload,
  );
}
