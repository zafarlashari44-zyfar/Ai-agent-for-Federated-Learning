export type AamiClass = "N" | "S" | "V" | "F" | "Q";

export type AttributionMethod =
  | "grad-cam"
  | "integrated-gradients"
  | "shap"
  | "saliency"
  | "none";

export type EvidenceDirection =
  | "supports"
  | "contradicts"
  | "neutral"
  | "unavailable";

export type SignalQualityStatus =
  | "good"
  | "acceptable"
  | "poor"
  | "unusable";

export type ConsistencyStatus =
  | "strongly_supported"
  | "partially_supported"
  | "insufficient_evidence"
  | "conflicting_evidence"
  | "low_signal_quality"
  | "out_of_scope";

export interface ECGDatasetMetadata {
  datasetName: string;
  recordId: string;
  sourceFormat: string;
  sourceFileNames: string[];
  patientExternalId?: string;
  leadNames: string[];
  selectedLead: string;
  samplingRateHz: number;
  durationSeconds: number;
  totalSamples: number;
  preprocessingVersion?: string;
  modelVersion?: string;
  checkpointHash?: string;
}

export interface ECGSignalPoint {
  sampleIndex: number;
  timeSeconds: number;
  amplitude: number;
}

export interface ECGSignalWindow {
  startSample: number;
  endSample: number;
  startTimeSeconds: number;
  endTimeSeconds: number;
  points: ECGSignalPoint[];
}

export interface RPeak {
  index: number;
  sampleIndex: number;
  timeSeconds: number;
  amplitude?: number;
  confidence?: number;
}

export interface AttributionPoint {
  sampleIndex: number;
  timeSeconds: number;
  intensity: number;
}

export interface AttributionRegion {
  id: string;
  method: AttributionMethod;
  startSample: number;
  endSample: number;
  startTimeSeconds: number;
  endTimeSeconds: number;
  intensity: number;
  label?: string;
  description?: string;
}

export interface ClassProbability {
  classCode: AamiClass;
  className: string;
  probability: number;
}

export interface BeatPrediction {
  classCode: AamiClass;
  className: string;
  confidence: number;
  classProbabilities?: ClassProbability[];
}

export interface SegmentedBeat {
  beatIndex: number;
  rPeakSample: number;
  rPeakTimeSeconds: number;
  startSample: number;
  endSample: number;
  startTimeSeconds: number;
  endTimeSeconds: number;
  waveform?: ECGSignalPoint[];
  prediction: BeatPrediction;
  attributionRegions?: AttributionRegion[];
  isAbnormal: boolean;
}

export interface RecordingPrediction {
  classCode: AamiClass;
  className: string;
  confidence: number;
  dominantClass: AamiClass;
  abnormalBeatCount: number;
  abnormalBeatPercentage: number;
  totalBeatCount: number;
  classCounts: Record<AamiClass, number>;
  classProbabilities?: ClassProbability[];
}

export interface ClinicalEvidenceItem {
  id: string;
  featureName: string;
  observedValue: string | number | null;
  expectedRange?: string;
  direction: EvidenceDirection;
  explanation: string;
  confidence?: number;
  beatIndex?: number;
  startTimeSeconds?: number;
  endTimeSeconds?: number;
}

export interface SignalQualityAssessment {
  status: SignalQualityStatus;
  score: number;
  explanation?: string;
  warnings: string[];
}

export interface ECGUncertainty {
  predictionUncertainty: number;
  reasoningConfidence: number;
  consistencyStatus: ConsistencyStatus;
  limitations: string[];
  warnings: string[];
}

export interface ECGReasoningResult {
  conclusion: string;
  evidence: ClinicalEvidenceItem[];
  consistencyStatus: ConsistencyStatus;
  reasoningConfidence: number;
  limitations: string[];
  ruleTrace?: string[];
  reasoningVersion?: string;
}

export interface ECGAnalysisResult {
  analysisId: string;
  generatedAt: string;
  metadata: ECGDatasetMetadata;
  signalQuality: SignalQualityAssessment;
  waveform: ECGSignalWindow;
  rPeaks: RPeak[];
  beats: SegmentedBeat[];
  attributionMethod: AttributionMethod;
  attributionPoints?: AttributionPoint[];
  attributionRegions: AttributionRegion[];
  prediction: RecordingPrediction;
  reasoning: ECGReasoningResult;
  uncertainty: ECGUncertainty;
}

export interface ECGUploadRequestMetadata {
  patientExternalId?: string;
  datasetName?: string;
  recordId?: string;
  selectedLead?: string;
  samplingRateHz?: number;
}

export interface ECGAnalysisApiError {
  error: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface ECGAnalysisApiResponse {
  success: boolean;
  result?: ECGAnalysisResult;
  error?: ECGAnalysisApiError;
}
