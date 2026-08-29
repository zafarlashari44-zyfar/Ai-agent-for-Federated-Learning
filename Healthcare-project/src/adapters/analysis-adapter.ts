import type {
  AnalysisResponse,
  EvidenceItemResponse,
} from "@/types/backend-analysis";

import type {
  AamiClass,
  AttributionMethod,
  AttributionRegion,
  ClinicalEvidenceItem,
  ConsistencyStatus,
  ECGAnalysisApiResponse,
  ECGAnalysisResult,
  ECGSignalPoint,
  EvidenceDirection,
  SegmentedBeat,
} from "@/types/ecg-analysis";

const CLASS_CODES: AamiClass[] = ["N", "S", "V", "F", "Q"];

const CLASS_NAMES: Record<AamiClass, string> = {
  N: "Normal",
  S: "Supraventricular ectopic beat",
  V: "Ventricular ectopic beat",
  F: "Fusion beat",
  Q: "Unknown or unclassifiable beat",
};

function isRecord(
  value: unknown,
): value is Record<string, unknown> {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value)
  );
}

function isBackendResponse(
  value: unknown,
): value is AnalysisResponse {
  if (!isRecord(value)) return false;

  return (
    isRecord(value.signal) &&
    isRecord(value.waveform) &&
    isRecord(value.prediction) &&
    isRecord(value.recording_summary) &&
    isRecord(value.evidence) &&
    isRecord(value.reasoning) &&
    isRecord(value.clinical_report) &&
    Array.isArray(value.analysis_warnings)
  );
}

function classCodeFromIndex(index: number): AamiClass {
  return CLASS_CODES[index] ?? "Q";
}

function mapConsistencyStatus(
  value: AnalysisResponse["reasoning"]["consistency_status"],
): ConsistencyStatus {
  switch (value) {
    case "strongly_supported":
      return "strongly_supported";

    case "partially_supported":
      return "partially_supported";

    case "conflicting":
      return "conflicting_evidence";

    case "insufficient_evidence":
    default:
      return "insufficient_evidence";
  }
}

function mapEvidenceDirection(
  value: EvidenceItemResponse["direction"],
): EvidenceDirection {
  switch (value) {
    case "supporting":
      return "supports";

    case "conflicting":
      return "contradicts";

    case "neutral":
    default:
      return "neutral";
  }
}

function mapEvidenceItem(
  item: EvidenceItemResponse,
): ClinicalEvidenceItem {
  return {
    id: item.evidence_id,
    featureName: item.feature_name,
    observedValue: item.measured_value,
    direction: mapEvidenceDirection(item.direction),
    explanation: item.interpretation,
    confidence: item.reliability,
  };
}

function buildWaveformPoints(
  response: AnalysisResponse,
): ECGSignalPoint[] {
  return response.waveform.sample_indices.map(
    (sampleIndex, index) => ({
      sampleIndex,
      timeSeconds:
        response.waveform.timestamps_seconds[index] ??
        sampleIndex / response.waveform.sampling_rate_hz,
      amplitude: response.waveform.amplitudes[index] ?? 0,
    }),
  );
}

function mapAttributionMethod(
  methodId: string,
): AttributionMethod {
  if (
    methodId === "grad-cam" ||
    methodId === "grad-cam-1d"
  ) {
    return "grad-cam";
  }

  if (methodId === "integrated-gradients") {
    return "integrated-gradients";
  }

  return "saliency";
}


function buildAttributionRegions(
  response: AnalysisResponse,
): AttributionRegion[] {
  const overlay = response.recording_attribution_overlay;

  if (!overlay || overlay.sample_indices.length === 0) {
    return [];
  }

  const safeOverlay = overlay;

  const method = mapAttributionMethod(
    safeOverlay.method_id,
  );

  const regions: AttributionRegion[] = [];
  const threshold = 0.35;

  let startIndex: number | null = null;
  let maximumIntensity = 0;

  function closeRegion(stopIndex: number) {
    if (startIndex === null) return;

    const startSample =
      safeOverlay.sample_indices[startIndex] ?? 0;

    const endSample =
      safeOverlay.sample_indices[stopIndex] ?? startSample;

    const startTime =
      safeOverlay.timestamps_seconds[startIndex] ??
      startSample / safeOverlay.sampling_rate_hz;

    const endTime =
      safeOverlay.timestamps_seconds[stopIndex] ??
      endSample / safeOverlay.sampling_rate_hz;

    regions.push({
      id: `attribution-${regions.length + 1}`,
      method: method as AttributionMethod,
      startSample,
      endSample,
      startTimeSeconds: startTime,
      endTimeSeconds: endTime,
      intensity: maximumIntensity,
      label: "Model attribution region",
      description:
        "This region contributed to the model prediction.",
    });

    startIndex = null;
    maximumIntensity = 0;
  }

  safeOverlay.maximum_attributions.forEach(
    (intensity, index) => {
      if (intensity >= threshold) {
        if (startIndex === null) {
          startIndex = index;
        }

        maximumIntensity = Math.max(
          maximumIntensity,
          intensity,
        );

        return;
      }

      if (startIndex !== null) {
        closeRegion(Math.max(index - 1, startIndex));
      }
    },
  );

  if (startIndex !== null) {
    closeRegion(
      safeOverlay.maximum_attributions.length - 1,
    );
  }

  return regions;
}

function buildSegmentedBeats(
  response: AnalysisResponse,
): SegmentedBeat[] {
  const waveformPoints = buildWaveformPoints(response);

  const explainedBeats = new Map(
    response.recording_explanation?.beat_explanations.map(
      (beat) => [beat.beat_index, beat],
    ) ?? [],
  );

  return response.recording_summary.beat_results.map(
    (beat): SegmentedBeat => {
      const classCode = classCodeFromIndex(
        beat.prediction.predicted_class,
      );

      const explanation = explainedBeats.get(
        beat.beat_index,
      );

      const attributionRegions =
        explanation?.attribution_maps.flatMap(
          (attributionMap, mapIndex) => {
            if (attributionMap.points.length === 0) {
              return [];
            }

            const intensities =
              attributionMap.points.map((point) =>
                Math.abs(point.attribution),
              );

            return [
              {
                id: `beat-${beat.beat_index}-map-${mapIndex}`,
                method: mapAttributionMethod(
                  attributionMap.method_id,
                ),
                startSample:
                  attributionMap.source_start_sample_index,
                endSample:
                  attributionMap.source_stop_sample_index_exclusive -
                  1,
                startTimeSeconds:
                  beat.source_start_timestamp_seconds,
                endTimeSeconds:
                  beat.source_stop_timestamp_seconds_exclusive,
                intensity: Math.max(...intensities),
                label: attributionMap.target_label,
                description: `${attributionMap.method_id} attribution for the predicted beat class.`,
              },
            ] satisfies AttributionRegion[];
          },
        ) ?? [];

     
      const waveform = waveformPoints.filter(
        (point) =>
          point.timeSeconds >=
            beat.source_start_timestamp_seconds &&
          point.timeSeconds <
            beat.source_stop_timestamp_seconds_exclusive,
      );

 return {
        beatIndex: beat.beat_index,
        rPeakSample: beat.r_peak_sample_index,
        rPeakTimeSeconds:
          beat.r_peak_timestamp_seconds,
        startSample:
          beat.source_start_sample_index,
        endSample:
          beat.source_stop_sample_index_exclusive - 1,
        startTimeSeconds:
          beat.source_start_timestamp_seconds,
        endTimeSeconds:
          beat.source_stop_timestamp_seconds_exclusive,
                waveform,
        prediction: {
          classCode,
          className:
            beat.prediction.predicted_label ||
            CLASS_NAMES[classCode],
          confidence: beat.prediction.confidence,
          classProbabilities:
            beat.prediction.probabilities.map(
              (probability, index) => {
                const code = classCodeFromIndex(index);

                return {
                  classCode: code,
                  className: CLASS_NAMES[code],
                  probability,
                };
              },
            ),
        },
        attributionRegions,
        isAbnormal: classCode !== "N",
      };
    },
  );
}

function adaptBackendResponse(
  response: AnalysisResponse,
): ECGAnalysisResult {
  const waveformPoints = buildWaveformPoints(response);
  const beats = buildSegmentedBeats(response);

  const recordingClassCode = classCodeFromIndex(
    response.prediction.predicted_class,
  );

  const allEvidence = [
    ...response.evidence.supporting,
    ...response.evidence.conflicting,
    ...response.evidence.neutral,
  ].map(mapEvidenceItem);

  const signalQuality =
    response.signal_suitability;

  const qualityStatus =
    signalQuality?.status === "rejected"
      ? "unusable"
      : signalQuality?.status ===
          "accepted_with_warnings"
        ? "acceptable"
        : signalQuality
          ? "good"
          : "poor";

  const consistencyStatus = mapConsistencyStatus(
    response.reasoning.consistency_status,
  );

  return {
    analysisId: crypto.randomUUID(),
    generatedAt: new Date().toISOString(),

    metadata: {
      datasetName: response.signal.source,
      recordId: response.signal.record_id,
      sourceFormat:
        response.signal.source_format ?? "unknown",
      sourceFileNames: [],
      leadNames:
        response.signal.lead_names.length > 0
          ? response.signal.lead_names
          : response.signal.lead_name
            ? [response.signal.lead_name]
            : [],
      selectedLead:
        response.signal.lead_name ?? "Unknown lead",
      samplingRateHz:
        response.signal.sampling_rate_hz,
      durationSeconds:
        response.signal.duration_seconds,
      totalSamples: response.signal.sample_count,
      preprocessingVersion:
        response.prediction.preprocessing_version,
      modelVersion:
        response.prediction.model_version,
      checkpointHash:
        response.prediction.checkpoint_hash,
    },

    signalQuality: {
      status: qualityStatus,
      score: signalQuality?.quality_score ?? 0,
      explanation:
        signalQuality?.suitable_for_processing
          ? "The signal passed the suitability assessment."
          : "The signal did not pass the suitability assessment.",
      warnings: [
        ...(signalQuality?.warnings ?? []),
        ...(signalQuality?.rejection_reasons ?? []),
      ],
    },

    waveform: {
      startSample:
        response.waveform.start_sample_index,
      endSample:
        response.waveform.stop_sample_index_exclusive -
        1,
      startTimeSeconds:
        response.waveform.timestamps_seconds[0] ?? 0,
      endTimeSeconds:
        response.waveform.timestamps_seconds.at(-1) ??
        response.signal.duration_seconds,
      points: waveformPoints,
    },

    rPeaks: beats.map((beat, index) => ({
      index,
      sampleIndex: beat.rPeakSample,
      timeSeconds: beat.rPeakTimeSeconds,
      confidence: beat.prediction.confidence,
    })),

    beats,

    attributionMethod:
      (response.recording_attribution_overlay
        ?.method_id as AttributionMethod) ?? "none",

    attributionRegions:
      buildAttributionRegions(response),

    prediction: {
      classCode: recordingClassCode,
      className:
        response.prediction.predicted_label ||
        CLASS_NAMES[recordingClassCode],
      confidence: response.prediction.confidence,
      dominantClass: classCodeFromIndex(
        response.recording_summary
          .dominant_predicted_class,
      ),
      abnormalBeatCount:
        response.recording_summary
          .abnormal_beat_count,
      abnormalBeatPercentage:
        response.recording_summary
          .abnormal_beat_percentage,
      totalBeatCount:
        response.recording_summary.total_valid_beats,
      classCounts: {
        N:
          response.recording_summary.class_counts.N ??
          0,
        S:
          response.recording_summary.class_counts.S ??
          0,
        V:
          response.recording_summary.class_counts.V ??
          0,
        F:
          response.recording_summary.class_counts.F ??
          0,
        Q:
          response.recording_summary.class_counts.Q ??
          0,
      },
      classProbabilities:
        response.prediction.probabilities.map(
          (probability, index) => {
            const code = classCodeFromIndex(index);

            return {
              classCode: code,
              className: CLASS_NAMES[code],
              probability,
            };
          },
        ),
    },

    reasoning: {
      conclusion: response.reasoning.conclusion,
      evidence: allEvidence,
      consistencyStatus,
      reasoningConfidence:
        response.reasoning.reasoning_confidence,
      limitations: [
        ...response.reasoning.limitations,
        ...response.evidence.limitations,
        response.model_scope_statement,
      ],
      ruleTrace: response.reasoning.rule_trace,
      reasoningVersion:
        response.reasoning.reasoning_version,
    },

    uncertainty: {
      predictionUncertainty:
        1 - response.prediction.confidence,
      reasoningConfidence:
        response.reasoning.reasoning_confidence,
      consistencyStatus,
      limitations: [
        ...response.clinical_report.limitations,
        response.recommended_interpretation,
      ],
      warnings: [
        ...response.analysis_warnings,
        ...(response.ood_assessment?.warnings ?? []),
        ...(response.ood_assessment?.reasons ?? []),
      ],
    },
  };
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
  if (isBackendResponse(payload)) {
    return adaptBackendResponse(payload);
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

    if (isBackendResponse(wrapped.result)) {
      return adaptBackendResponse(wrapped.result);
    }

    for (const key of [
      "analysis",
      "analysis_result",
      "result",
      "data",
    ]) {
      const candidate = payload[key];

      if (isBackendResponse(candidate)) {
        return adaptBackendResponse(candidate);
      }
    }
  }

  throw new ECGAnalysisAdapterError(
    "The ECG analysis response does not match the FastAPI contract.",
    payload,
  );
}




