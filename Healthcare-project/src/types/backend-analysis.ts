export type AnalysisScope =
  | "validated_mit_bih_compatible"
  | "exploratory_external_source"
  | "unsupported";

export type ConsistencyStatus =
  | "strongly_supported"
  | "partially_supported"
  | "conflicting"
  | "insufficient_evidence";

export type EvidenceDirection =
  | "supporting"
  | "conflicting"
  | "neutral";

export type SignalSuitabilityStatus =
  | "accepted"
  | "accepted_with_warnings"
  | "rejected";

export type OODStatus =
  | "in_distribution_like"
  | "uncertain"
  | "likely_out_of_distribution";

export interface SourceSignalMetadataResponse {
  record_id: string;
  source_format: string | null;
  original_sampling_rate_hz: number | null;
  original_units: string | null;
  original_sample_count: number | null;
  original_duration_seconds: number | null;
  available_lead_names: string[];
  selected_lead: string | null;
  warnings: string[];
}

export interface CompactSignalWaveformResponse {
  record_id: string;
  sampling_rate_hz: number;
  total_sample_count: number;
  start_sample_index: number;
  stop_sample_index_exclusive: number;
  source_window_sample_count: number;
  returned_point_count: number;
  downsampled: boolean;
  downsampling_strategy: string | null;
  sample_indices: number[];
  timestamps_seconds: number[];
  amplitudes: number[];
  units: string | null;
  lead_name: string | null;
  warnings: string[];
}

export interface HarmonisationMetadataResponse {
  target_sampling_rate_hz: number | null;
  target_units: string | null;
  resampled: boolean;
  unit_conversion_applied: string | null;
  resampling_method: string | null;
  resampling_up_factor: number | null;
  resampling_down_factor: number | null;
  harmonised_sample_count: number | null;
  harmonised_duration_seconds: number | null;
  transformations: string[];
  warnings: string[];
}

export interface SignalResponse {
  record_id: string;
  sampling_rate_hz: number;
  sample_count: number;
  duration_seconds: number;
  source: string;
  lead_name: string | null;
  source_format: string | null;
  original_sampling_rate_hz: number | null;
  lead_names: string[];
  units: string | null;
  original_sample_count: number | null;
  original_duration_seconds: number | null;
  warnings: string[];
  source_metadata: SourceSignalMetadataResponse;
  harmonisation_metadata: HarmonisationMetadataResponse;
}

export interface PredictionResponse {
  predicted_class: number;
  predicted_label: string;
  probabilities: number[];
  confidence: number;
  checkpoint_hash: string;
  model_version: string;
  preprocessing_version: string;
}

export interface BeatAnalysisResponse {
  beat_index: number;
  r_peak_sample_index: number;
  source_start_sample_index: number;
  source_stop_sample_index_exclusive: number;
  r_peak_timestamp_seconds: number;
  source_start_timestamp_seconds: number;
  source_stop_timestamp_seconds_exclusive: number;
  sampling_rate_hz: number;
  prediction: PredictionResponse;
}

export interface RecordingAnalysisSummaryResponse {
  total_valid_beats: number;
  class_counts: Record<string, number>;
  abnormal_beat_count: number;
  abnormal_beat_percentage: number;
  dominant_predicted_class: number;
  dominant_predicted_label: string;
  beat_results: BeatAnalysisResponse[];
}

export interface AttributionPointResponse {
  beat_sample_index: number;
  source_sample_index: number;
  timestamp_seconds: number;
  attribution: number;
  input_value: number;
}

export interface AttributionMapResponse {
  method_id: string;
  method_version: string;
  target_class: number;
  target_label: string;
  target_output: string;
  points: AttributionPointResponse[];
  signed: boolean;
  native_resolution: number;
  interpolation_method: string | null;
  normalisation: string;
  sampling_rate_hz: number;
  source_start_sample_index: number;
  source_stop_sample_index_exclusive: number;
  convergence_delta: number | null;
  parameters: [string, string][];
  warnings: string[];
}

export interface BeatExplanationResponse {
  beat_index: number;
  r_peak_sample_index: number;
  r_peak_timestamp_seconds: number;
  source_start_sample_index: number;
  source_stop_sample_index_exclusive: number;
  source_start_timestamp_seconds: number;
  source_stop_timestamp_seconds_exclusive: number;
  sampling_rate_hz: number;
  predicted_class: number;
  predicted_label: string;
  prediction_confidence: number;
  attribution_maps: AttributionMapResponse[];
  warnings: string[];
}

export interface RecordingExplanationResponse {
  record_id: string;
  selection_policy: string;
  total_valid_beats: number;
  total_explained_beats: number;
  beat_explanations: BeatExplanationResponse[];
  requested_methods: string[];
  completed_methods: string[];
  model_version: string;
  checkpoint_hash: string;
  preprocessing_version: string;
  warnings: string[];
}

export interface CompactRecordingAttributionOverlayResponse {
  record_id: string;
  method_id: string;
  method_version: string;
  sampling_rate_hz: number;
  total_source_samples: number;
  aggregation_method: string;
  start_sample_index: number;
  stop_sample_index_exclusive: number;
  source_window_sample_count: number;
  returned_point_count: number;
  downsampled: boolean;
  downsampling_strategy: string | null;
  sample_indices: number[];
  timestamps_seconds: number[];
  maximum_attributions: number[];
  mean_attributions: number[];
  coverage_counts: number[];
  contributing_beat_indices: number[][];
  explained_beat_count: number;
  warnings: string[];
}

export interface EvidenceItemResponse {
  evidence_id: string;
  feature_name: string;
  measured_value: number | string | null;
  unit: string | null;
  interpretation: string;
  direction: EvidenceDirection;
  reliability: number;
  source_reference: string;
}

export interface EvidenceResponse {
  supporting: EvidenceItemResponse[];
  conflicting: EvidenceItemResponse[];
  neutral: EvidenceItemResponse[];
  limitations: string[];
  evidence_version: string;
}

export interface ReasoningResponse {
  consistency_status: ConsistencyStatus;
  reasoning_confidence: number;
  conclusion: string;
  limitations: string[];
  rule_trace: string[];
  reasoning_version: string;
}

export interface ClinicalReportResponse {
  record_id: string;
  predicted_label: string;
  prediction_confidence: number;
  consistency_status: ConsistencyStatus;
  reasoning_confidence: number;
  summary: string;
  supporting_findings: string[];
  conflicting_findings: string[];
  limitations: string[];
  recommended_action: string;
  model_version: string;
  preprocessing_version: string;
  evidence_version: string;
  reasoning_version: string;
  report_version: string;
  disclaimer: string;
}

export interface NarrativeResponse {
  doctor_report: string;
  next_of_kin_summary: string;
  provider: string;
  model_name: string;
  prompt_version: string;
  fallback_used: boolean;
  warnings: string[];
}

export interface SignalSuitabilityResponse {
  status: SignalSuitabilityStatus;
  suitable_for_processing: boolean;
  quality_score: number;
  duration_seconds: number;
  sampling_rate_hz: number;
  selected_lead: string | null;
  units: string | null;
  detected_r_peak_count: number;
  estimated_heart_rate_bpm: number | null;
  finite_sample_ratio: number;
  flatline_percentage: number;
  clipping_percentage: number;
  noise_score: number;
  warnings: string[];
  rejection_reasons: string[];
}

export interface OODAssessmentResponse {
  status: OODStatus;
  heuristic_score: number;
  maximum_class_probability: number;
  normalized_prediction_entropy: number;
  q_class_proportion: number;
  low_confidence_beat_proportion: number;
  probability_instability: number;
  indicators: string[];
  reasons: string[];
  warnings: string[];
}

export interface AnalysisResponse {
  signal: SignalResponse;
  waveform: CompactSignalWaveformResponse;
  prediction: PredictionResponse;
  recording_summary: RecordingAnalysisSummaryResponse;
  evidence: EvidenceResponse;
  reasoning: ReasoningResponse;
  clinical_report: ClinicalReportResponse;
  narrative: NarrativeResponse;
  input_accepted: boolean;
  model_prediction_produced: boolean;
  signal_suitability: SignalSuitabilityResponse | null;
  ood_assessment: OODAssessmentResponse | null;
  analysis_scope: AnalysisScope;
  model_scope_statement: string;
  recommended_interpretation: string;
  analysis_warnings: string[];
  recording_explanation: RecordingExplanationResponse | null;
  recording_attribution_overlay:
    | CompactRecordingAttributionOverlayResponse
    | null;
}
export interface APIErrorResponse {
  error: string;
  detail: string;
}