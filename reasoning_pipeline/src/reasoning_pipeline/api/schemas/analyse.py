from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from reasoning_pipeline.domain.enums.statuses import (
    AnalysisScope,
    ConsistencyStatus,
    EvidenceDirection,
    OODStatus,
    SignalSuitabilityStatus,
)
from reasoning_pipeline.domain.models.attribution_map import AttributionMap
from reasoning_pipeline.domain.models.attribution_point import AttributionPoint
from reasoning_pipeline.domain.models.beat_analysis_result import (
    BeatAnalysisResult,
)
from reasoning_pipeline.domain.models.beat_explanation import BeatExplanation
from reasoning_pipeline.domain.models.clinical_report import ClinicalReport
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.domain.models.evidence_item import EvidenceItem
from reasoning_pipeline.domain.models.model_prediction import ModelPrediction
from reasoning_pipeline.domain.models.narrative_result import NarrativeResult
from reasoning_pipeline.domain.models.reasoning_result import ReasoningResult
from reasoning_pipeline.domain.models.recording_analysis_summary import (
    RecordingAnalysisSummary,
)
from reasoning_pipeline.domain.models.recording_attribution_overlay import (
    RecordingAttributionOverlay,
)
from reasoning_pipeline.domain.models.recording_attribution_point import (
    RecordingAttributionPoint,
)
from reasoning_pipeline.domain.models.recording_explanation import (
    RecordingExplanation,
)
from reasoning_pipeline.orchestration.analysis_result import ECGAnalysisResult


class SourceSignalMetadataResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    source_format: str | None
    original_sampling_rate_hz: float | None
    original_units: str | None
    original_sample_count: int | None
    original_duration_seconds: float | None
    available_lead_names: tuple[str, ...]
    selected_lead: str | None
    warnings: tuple[str, ...]


class HarmonisationMetadataResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_sampling_rate_hz: float | None
    target_units: str | None
    resampled: bool
    unit_conversion_applied: str | None
    resampling_method: str | None
    resampling_up_factor: int | None
    resampling_down_factor: int | None
    harmonised_sample_count: int | None
    harmonised_duration_seconds: float | None
    transformations: tuple[str, ...]
    warnings: tuple[str, ...]


class SignalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    sampling_rate_hz: float
    sample_count: int
    duration_seconds: float
    source: str
    lead_name: str | None
    source_format: str | None
    original_sampling_rate_hz: float | None
    lead_names: tuple[str, ...]
    units: str | None
    original_sample_count: int | None
    original_duration_seconds: float | None
    warnings: tuple[str, ...]
    source_metadata: SourceSignalMetadataResponse
    harmonisation_metadata: HarmonisationMetadataResponse

    @classmethod
    def from_domain(cls, signal: ECGSignal) -> SignalResponse:
        return cls(
            record_id=signal.record_id,
            sampling_rate_hz=signal.sampling_rate_hz,
            sample_count=signal.sample_count,
            duration_seconds=signal.duration_seconds,
            source=signal.source,
            lead_name=signal.lead_name,
            source_format=signal.source_format,
            original_sampling_rate_hz=signal.original_sampling_rate_hz,
            lead_names=signal.lead_names,
            units=signal.units,
            original_sample_count=signal.original_sample_count,
            original_duration_seconds=signal.original_duration_seconds,
            warnings=signal.warnings,
            source_metadata=SourceSignalMetadataResponse(
                record_id=signal.record_id,
                source_format=signal.source_format,
                original_sampling_rate_hz=signal.original_sampling_rate_hz,
                original_units=signal.original_units,
                original_sample_count=signal.original_sample_count,
                original_duration_seconds=signal.original_duration_seconds,
                available_lead_names=signal.lead_names,
                selected_lead=signal.lead_name,
                warnings=signal.warnings,
            ),
            harmonisation_metadata=HarmonisationMetadataResponse(
                target_sampling_rate_hz=signal.target_sampling_rate_hz,
                target_units=signal.target_units,
                resampled=signal.resampled,
                unit_conversion_applied=signal.unit_conversion_applied,
                resampling_method=signal.resampling_method,
                resampling_up_factor=signal.resampling_up_factor,
                resampling_down_factor=signal.resampling_down_factor,
                harmonised_sample_count=signal.harmonised_sample_count,
                harmonised_duration_seconds=(
                    signal.harmonised_duration_seconds
                ),
                transformations=signal.harmonisation_transformations,
                warnings=signal.harmonisation_warnings,
            ),
        )



class CompactSignalWaveformResponse(BaseModel):
    """Compact frontend representation of the harmonised ECG waveform."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    sampling_rate_hz: float
    total_sample_count: int
    start_sample_index: int
    stop_sample_index_exclusive: int
    source_window_sample_count: int
    returned_point_count: int
    downsampled: bool
    downsampling_strategy: str | None
    sample_indices: tuple[int, ...]
    timestamps_seconds: tuple[float, ...]
    amplitudes: tuple[float, ...]
    units: str | None
    lead_name: str | None
    warnings: tuple[str, ...]

    @classmethod
    def from_domain(
        cls,
        signal: ECGSignal,
        *,
        start_sample: int | None = None,
        stop_sample: int | None = None,
        downsample_limit: int | None = None,
    ) -> "CompactSignalWaveformResponse":
        start = 0 if start_sample is None else start_sample
        stop = signal.sample_count if stop_sample is None else stop_sample

        cls._validate_window(
            total_sample_count=signal.sample_count,
            start_sample=start,
            stop_sample=stop,
            downsample_limit=downsample_limit,
        )

        indexed_samples = tuple(
            (index, signal.samples[index])
            for index in range(start, stop)
        )

        selected_samples = indexed_samples
        downsampled = (
            downsample_limit is not None
            and len(indexed_samples) > downsample_limit
        )

        if downsampled:
            assert downsample_limit is not None
            selected_samples = cls._min_max_downsample(
                indexed_samples,
                limit=downsample_limit,
            )

        return cls(
            record_id=signal.record_id,
            sampling_rate_hz=signal.sampling_rate_hz,
            total_sample_count=signal.sample_count,
            start_sample_index=start,
            stop_sample_index_exclusive=stop,
            source_window_sample_count=stop - start,
            returned_point_count=len(selected_samples),
            downsampled=downsampled,
            downsampling_strategy=(
                "contiguous-bin-min-max"
                if downsampled
                else None
            ),
            sample_indices=tuple(
                index for index, _ in selected_samples
            ),
            timestamps_seconds=tuple(
                index / signal.sampling_rate_hz
                for index, _ in selected_samples
            ),
            amplitudes=tuple(
                float(value) for _, value in selected_samples
            ),
            units=signal.units,
            lead_name=signal.lead_name,
            warnings=signal.warnings,
        )

    @staticmethod
    def _validate_window(
        *,
        total_sample_count: int,
        start_sample: int,
        stop_sample: int,
        downsample_limit: int | None,
    ) -> None:
        if start_sample < 0:
            raise ValueError(
                "waveform_start_sample must be non-negative."
            )

        if stop_sample > total_sample_count:
            raise ValueError(
                "waveform_stop_sample cannot exceed the recording length."
            )

        if start_sample >= stop_sample:
            raise ValueError(
                "waveform_start_sample must be less than waveform_stop_sample."
            )

        if downsample_limit is not None and downsample_limit < 2:
            raise ValueError(
                "waveform_downsample_limit must be at least two."
            )

    @staticmethod
    def _min_max_downsample(
        samples: tuple[tuple[int, float], ...],
        *,
        limit: int,
    ) -> tuple[tuple[int, float], ...]:
        if len(samples) <= limit:
            return samples

        bin_count = max(1, limit // 2)
        selected: list[tuple[int, float]] = []

        for bin_index in range(bin_count):
            bin_start = (bin_index * len(samples)) // bin_count
            bin_stop = ((bin_index + 1) * len(samples)) // bin_count
            bin_samples = samples[bin_start:bin_stop]

            if not bin_samples:
                continue

            minimum = min(bin_samples, key=lambda item: item[1])
            maximum = max(bin_samples, key=lambda item: item[1])

            if minimum[0] <= maximum[0]:
                selected.extend((minimum, maximum))
            else:
                selected.extend((maximum, minimum))

        unique = {
            index: (index, value)
            for index, value in selected
        }

        return tuple(
            unique[index]
            for index in sorted(unique)
        )[:limit]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    predicted_class: int
    predicted_label: str
    probabilities: tuple[float, ...]
    confidence: float = Field(ge=0.0, le=1.0)
    checkpoint_hash: str
    model_version: str
    preprocessing_version: str

    @classmethod
    def from_domain(
        cls,
        prediction: ModelPrediction,
    ) -> PredictionResponse:
        return cls(
            predicted_class=prediction.predicted_class,
            predicted_label=prediction.predicted_label,
            probabilities=prediction.probabilities,
            confidence=prediction.confidence,
            checkpoint_hash=prediction.checkpoint_hash,
            model_version=prediction.model_version,
            preprocessing_version=prediction.preprocessing_version,
        )


class BeatAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    beat_index: int
    r_peak_sample_index: int
    source_start_sample_index: int
    source_stop_sample_index_exclusive: int
    r_peak_timestamp_seconds: float
    source_start_timestamp_seconds: float
    source_stop_timestamp_seconds_exclusive: float
    sampling_rate_hz: float
    prediction: PredictionResponse

    @classmethod
    def from_domain(
        cls,
        result: BeatAnalysisResult,
    ) -> BeatAnalysisResponse:
        return cls(
            beat_index=result.beat_index,
            r_peak_sample_index=result.r_peak_sample_index,
            source_start_sample_index=result.source_start_sample_index,
            source_stop_sample_index_exclusive=(
                result.source_stop_sample_index_exclusive
            ),
            r_peak_timestamp_seconds=result.r_peak_timestamp_seconds,
            source_start_timestamp_seconds=(
                result.source_start_timestamp_seconds
            ),
            source_stop_timestamp_seconds_exclusive=(
                result.source_stop_timestamp_seconds_exclusive
            ),
            sampling_rate_hz=result.sampling_rate_hz,
            prediction=PredictionResponse.from_domain(result.prediction),
        )


class RecordingAnalysisSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    total_valid_beats: int
    class_counts: dict[str, int]
    abnormal_beat_count: int
    abnormal_beat_percentage: float
    dominant_predicted_class: int
    dominant_predicted_label: str
    beat_results: tuple[BeatAnalysisResponse, ...]

    @classmethod
    def from_domain(
        cls,
        summary: RecordingAnalysisSummary,
    ) -> RecordingAnalysisSummaryResponse:
        return cls(
            total_valid_beats=summary.total_valid_beats,
            class_counts=dict(summary.class_counts),
            abnormal_beat_count=summary.abnormal_beat_count,
            abnormal_beat_percentage=summary.abnormal_beat_percentage,
            dominant_predicted_class=summary.dominant_predicted_class,
            dominant_predicted_label=summary.dominant_predicted_label,
            beat_results=tuple(
                BeatAnalysisResponse.from_domain(result)
                for result in summary.beat_results
            ),
        )


class AttributionPointResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    beat_sample_index: int
    source_sample_index: int
    timestamp_seconds: float
    attribution: float
    input_value: float

    @classmethod
    def from_domain(cls, point: AttributionPoint) -> AttributionPointResponse:
        return cls(
            beat_sample_index=point.beat_sample_index,
            source_sample_index=point.source_sample_index,
            timestamp_seconds=point.timestamp_seconds,
            attribution=point.attribution,
            input_value=point.input_value,
        )


class AttributionMapResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method_id: str
    method_version: str
    target_class: int
    target_label: str
    target_output: str
    points: tuple[AttributionPointResponse, ...]
    signed: bool
    native_resolution: int
    interpolation_method: str | None
    normalisation: str
    sampling_rate_hz: float
    source_start_sample_index: int
    source_stop_sample_index_exclusive: int
    convergence_delta: float | None
    parameters: tuple[tuple[str, str], ...]
    warnings: tuple[str, ...]

    @classmethod
    def from_domain(cls, value: AttributionMap) -> AttributionMapResponse:
        return cls(
            method_id=value.method_id,
            method_version=value.method_version,
            target_class=value.target_class,
            target_label=value.target_label,
            target_output=value.target_output,
            points=tuple(
                AttributionPointResponse.from_domain(point)
                for point in value.points
            ),
            signed=value.signed,
            native_resolution=value.native_resolution,
            interpolation_method=value.interpolation_method,
            normalisation=value.normalisation,
            sampling_rate_hz=value.sampling_rate_hz,
            source_start_sample_index=value.source_start_sample_index,
            source_stop_sample_index_exclusive=(
                value.source_stop_sample_index_exclusive
            ),
            convergence_delta=value.convergence_delta,
            parameters=value.parameters,
            warnings=value.warnings,
        )


class BeatExplanationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    beat_index: int
    r_peak_sample_index: int
    r_peak_timestamp_seconds: float
    source_start_sample_index: int
    source_stop_sample_index_exclusive: int
    source_start_timestamp_seconds: float
    source_stop_timestamp_seconds_exclusive: float
    sampling_rate_hz: float
    predicted_class: int
    predicted_label: str
    prediction_confidence: float
    attribution_maps: tuple[AttributionMapResponse, ...]
    warnings: tuple[str, ...]

    @classmethod
    def from_domain(cls, value: BeatExplanation) -> BeatExplanationResponse:
        return cls(
            beat_index=value.beat_index,
            r_peak_sample_index=value.r_peak_sample_index,
            r_peak_timestamp_seconds=value.r_peak_timestamp_seconds,
            source_start_sample_index=value.source_start_sample_index,
            source_stop_sample_index_exclusive=(
                value.source_stop_sample_index_exclusive
            ),
            source_start_timestamp_seconds=(
                value.source_start_timestamp_seconds
            ),
            source_stop_timestamp_seconds_exclusive=(
                value.source_stop_timestamp_seconds_exclusive
            ),
            sampling_rate_hz=value.sampling_rate_hz,
            predicted_class=value.predicted_class,
            predicted_label=value.predicted_label,
            prediction_confidence=value.prediction_confidence,
            attribution_maps=tuple(
                AttributionMapResponse.from_domain(attribution_map)
                for attribution_map in value.attribution_maps
            ),
            warnings=value.warnings,
        )


class RecordingExplanationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    selection_policy: str
    total_valid_beats: int
    total_explained_beats: int
    beat_explanations: tuple[BeatExplanationResponse, ...]
    requested_methods: tuple[str, ...]
    completed_methods: tuple[str, ...]
    model_version: str
    checkpoint_hash: str
    preprocessing_version: str
    warnings: tuple[str, ...]

    @classmethod
    def from_domain(
        cls,
        value: RecordingExplanation,
    ) -> RecordingExplanationResponse:
        return cls(
            record_id=value.record_id,
            selection_policy=value.selection_policy,
            total_valid_beats=value.total_valid_beats,
            total_explained_beats=value.total_explained_beats,
            beat_explanations=tuple(
                BeatExplanationResponse.from_domain(beat)
                for beat in value.beat_explanations
            ),
            requested_methods=value.requested_methods,
            completed_methods=value.completed_methods,
            model_version=value.model_version,
            checkpoint_hash=value.checkpoint_hash,
            preprocessing_version=value.preprocessing_version,
            warnings=value.warnings,
        )


class RecordingAttributionPointResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_sample_index: int
    timestamp_seconds: float
    maximum_attribution: float
    mean_attribution: float
    coverage_count: int
    contributing_beat_indices: tuple[int, ...]

    @classmethod
    def from_domain(
        cls,
        point: RecordingAttributionPoint,
    ) -> RecordingAttributionPointResponse:
        return cls(
            source_sample_index=point.source_sample_index,
            timestamp_seconds=point.timestamp_seconds,
            maximum_attribution=point.maximum_attribution,
            mean_attribution=point.mean_attribution,
            coverage_count=point.coverage_count,
            contributing_beat_indices=point.contributing_beat_indices,
        )


class RecordingAttributionOverlayResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    method_id: str
    method_version: str
    sampling_rate_hz: float
    total_source_samples: int
    aggregation_method: str
    points: tuple[RecordingAttributionPointResponse, ...]
    explained_beat_count: int
    warnings: tuple[str, ...]

    @classmethod
    def from_domain(
        cls,
        overlay: RecordingAttributionOverlay,
    ) -> RecordingAttributionOverlayResponse:
        return cls(
            record_id=overlay.record_id,
            method_id=overlay.method_id,
            method_version=overlay.method_version,
            sampling_rate_hz=overlay.sampling_rate_hz,
            total_source_samples=overlay.total_source_samples,
            aggregation_method=overlay.aggregation_method,
            points=tuple(
                RecordingAttributionPointResponse.from_domain(point)
                for point in overlay.points
            ),
            explained_beat_count=overlay.explained_beat_count,
            warnings=overlay.warnings,
        )


class CompactRecordingAttributionOverlayResponse(BaseModel):
    """
    Frontend transport representation of a recording attribution overlay.

    Every position across the parallel arrays describes the same exact source
    sample. When downsampling is requested, each contiguous source window
    contributes the sample with the greatest maximum attribution.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    method_id: str
    method_version: str
    sampling_rate_hz: float
    total_source_samples: int
    aggregation_method: str
    start_sample_index: int
    stop_sample_index_exclusive: int
    source_window_sample_count: int
    returned_point_count: int
    downsampled: bool
    downsampling_strategy: str | None
    sample_indices: tuple[int, ...]
    timestamps_seconds: tuple[float, ...]
    maximum_attributions: tuple[float, ...]
    mean_attributions: tuple[float, ...]
    coverage_counts: tuple[int, ...]
    contributing_beat_indices: tuple[tuple[int, ...], ...]
    explained_beat_count: int
    warnings: tuple[str, ...]

    @classmethod
    def from_domain(
        cls,
        overlay: RecordingAttributionOverlay,
        *,
        start_sample: int | None = None,
        stop_sample: int | None = None,
        downsample_limit: int | None = None,
    ) -> CompactRecordingAttributionOverlayResponse:
        start = 0 if start_sample is None else start_sample
        stop = (
            overlay.total_source_samples
            if stop_sample is None
            else stop_sample
        )
        cls._validate_window(
            total_source_samples=overlay.total_source_samples,
            start_sample=start,
            stop_sample=stop,
            downsample_limit=downsample_limit,
        )

        window_points = overlay.points[start:stop]
        selected_points = window_points
        downsampled = (
            downsample_limit is not None
            and len(window_points) > downsample_limit
        )

        if downsampled:
            assert downsample_limit is not None
            selected_points = cls._maximum_preserving_downsample(
                window_points,
                limit=downsample_limit,
            )

        return cls(
            record_id=overlay.record_id,
            method_id=overlay.method_id,
            method_version=overlay.method_version,
            sampling_rate_hz=overlay.sampling_rate_hz,
            total_source_samples=overlay.total_source_samples,
            aggregation_method=overlay.aggregation_method,
            start_sample_index=start,
            stop_sample_index_exclusive=stop,
            source_window_sample_count=stop - start,
            returned_point_count=len(selected_points),
            downsampled=downsampled,
            downsampling_strategy=(
                "contiguous-bin-maximum-attribution"
                if downsampled
                else None
            ),
            sample_indices=tuple(
                point.source_sample_index for point in selected_points
            ),
            timestamps_seconds=tuple(
                point.timestamp_seconds for point in selected_points
            ),
            maximum_attributions=tuple(
                point.maximum_attribution for point in selected_points
            ),
            mean_attributions=tuple(
                point.mean_attribution for point in selected_points
            ),
            coverage_counts=tuple(
                point.coverage_count for point in selected_points
            ),
            contributing_beat_indices=tuple(
                point.contributing_beat_indices for point in selected_points
            ),
            explained_beat_count=overlay.explained_beat_count,
            warnings=overlay.warnings,
        )

    @staticmethod
    def _validate_window(
        *,
        total_source_samples: int,
        start_sample: int,
        stop_sample: int,
        downsample_limit: int | None,
    ) -> None:
        if start_sample < 0:
            raise ValueError("overlay_start_sample must be non-negative.")

        if stop_sample > total_source_samples:
            raise ValueError(
                "overlay_stop_sample cannot exceed the recording length."
            )

        if start_sample >= stop_sample:
            raise ValueError(
                "overlay_start_sample must be less than overlay_stop_sample."
            )

        if downsample_limit is not None and downsample_limit < 1:
            raise ValueError(
                "overlay_downsample_limit must be at least one."
            )

    @staticmethod
    def _maximum_preserving_downsample(
        points: tuple[RecordingAttributionPoint, ...],
        *,
        limit: int,
    ) -> tuple[RecordingAttributionPoint, ...]:
        point_count = len(points)
        selected: list[RecordingAttributionPoint] = []

        for bin_index in range(limit):
            bin_start = (bin_index * point_count) // limit
            bin_stop = ((bin_index + 1) * point_count) // limit
            bin_points = points[bin_start:bin_stop]
            selected.append(
                max(
                    bin_points,
                    key=lambda point: (
                        point.maximum_attribution,
                        -point.source_sample_index,
                    ),
                )
            )

        return tuple(selected)


class EvidenceItemResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    feature_name: str
    measured_value: float | int | str | None
    unit: str | None
    interpretation: str
    direction: EvidenceDirection
    reliability: float = Field(ge=0.0, le=1.0)
    source_reference: str

    @classmethod
    def from_domain(
        cls,
        item: EvidenceItem,
    ) -> EvidenceItemResponse:
        return cls(
            evidence_id=item.evidence_id,
            feature_name=item.feature_name,
            measured_value=item.measured_value,
            unit=item.unit,
            interpretation=item.interpretation,
            direction=item.direction,
            reliability=item.reliability,
            source_reference=item.source_reference,
        )


class EvidenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    supporting: tuple[EvidenceItemResponse, ...]
    conflicting: tuple[EvidenceItemResponse, ...]
    neutral: tuple[EvidenceItemResponse, ...]
    limitations: tuple[str, ...]
    evidence_version: str


class ReasoningResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    consistency_status: ConsistencyStatus
    reasoning_confidence: float = Field(ge=0.0, le=1.0)
    conclusion: str
    limitations: tuple[str, ...]
    rule_trace: tuple[str, ...]
    reasoning_version: str

    @classmethod
    def from_domain(
        cls,
        reasoning: ReasoningResult,
    ) -> ReasoningResponse:
        return cls(
            consistency_status=reasoning.consistency_status,
            reasoning_confidence=reasoning.reasoning_confidence,
            conclusion=reasoning.conclusion,
            limitations=reasoning.limitations,
            rule_trace=reasoning.rule_trace,
            reasoning_version=reasoning.reasoning_version,
        )


class ClinicalReportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: str
    predicted_label: str
    prediction_confidence: float = Field(ge=0.0, le=1.0)
    consistency_status: ConsistencyStatus
    reasoning_confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    supporting_findings: tuple[str, ...]
    conflicting_findings: tuple[str, ...]
    limitations: tuple[str, ...]
    recommended_action: str
    model_version: str
    preprocessing_version: str
    evidence_version: str
    reasoning_version: str
    report_version: str
    disclaimer: str

    @classmethod
    def from_domain(
        cls,
        report: ClinicalReport,
    ) -> ClinicalReportResponse:
        return cls(
            record_id=report.record_id,
            predicted_label=report.predicted_label,
            prediction_confidence=report.prediction_confidence,
            consistency_status=report.consistency_status,
            reasoning_confidence=report.reasoning_confidence,
            summary=report.summary,
            supporting_findings=report.supporting_findings,
            conflicting_findings=report.conflicting_findings,
            limitations=report.limitations,
            recommended_action=report.recommended_action,
            model_version=report.model_version,
            preprocessing_version=report.preprocessing_version,
            evidence_version=report.evidence_version,
            reasoning_version=report.reasoning_version,
            report_version=report.report_version,
            disclaimer=report.disclaimer,
        )


class NarrativeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    doctor_report: str
    next_of_kin_summary: str
    provider: str
    model_name: str
    prompt_version: str
    fallback_used: bool
    warnings: tuple[str, ...]

    @classmethod
    def from_domain(
        cls,
        narrative: NarrativeResult,
    ) -> NarrativeResponse:
        return cls(
            doctor_report=narrative.doctor_report,
            next_of_kin_summary=narrative.next_of_kin_summary,
            provider=narrative.provider,
            model_name=narrative.model_name,
            prompt_version=narrative.prompt_version,
            fallback_used=narrative.fallback_used,
            warnings=narrative.warnings,
        )


class SignalSuitabilityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: SignalSuitabilityStatus
    suitable_for_processing: bool
    quality_score: float
    duration_seconds: float
    sampling_rate_hz: float
    selected_lead: str | None
    units: str | None
    detected_r_peak_count: int
    estimated_heart_rate_bpm: float | None
    finite_sample_ratio: float
    flatline_percentage: float
    clipping_percentage: float
    noise_score: float
    warnings: tuple[str, ...]
    rejection_reasons: tuple[str, ...]


class OODAssessmentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: OODStatus
    heuristic_score: int
    maximum_class_probability: float
    normalized_prediction_entropy: float
    q_class_proportion: float
    low_confidence_beat_proportion: float
    probability_instability: float
    indicators: tuple[str, ...]
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    signal: SignalResponse
    waveform: CompactSignalWaveformResponse
    prediction: PredictionResponse
    recording_summary: RecordingAnalysisSummaryResponse
    evidence: EvidenceResponse
    reasoning: ReasoningResponse
    clinical_report: ClinicalReportResponse
    narrative: NarrativeResponse
    input_accepted: bool
    model_prediction_produced: bool
    signal_suitability: SignalSuitabilityResponse | None
    ood_assessment: OODAssessmentResponse | None
    analysis_scope: AnalysisScope
    model_scope_statement: str
    recommended_interpretation: str
    analysis_warnings: tuple[str, ...]
    recording_explanation: RecordingExplanationResponse | None = None
    recording_attribution_overlay: (
        CompactRecordingAttributionOverlayResponse | None
    ) = None

    @classmethod
    def from_domain(
        cls,
        result: ECGAnalysisResult,
        *,
        include_explanations: bool = True,
        include_overlay: bool = True,
        overlay_start_sample: int | None = None,
        overlay_stop_sample: int | None = None,
        overlay_downsample_limit: int | None = None,
        waveform_start_sample: int | None = None,
        waveform_stop_sample: int | None = None,
        waveform_downsample_limit: int | None = 20000,
    ) -> AnalysisResponse:
        evidence = result.evidence

        return cls(
            signal=SignalResponse.from_domain(result.signal),
            waveform=CompactSignalWaveformResponse.from_domain(
                result.signal,
                start_sample=waveform_start_sample,
                stop_sample=waveform_stop_sample,
                downsample_limit=waveform_downsample_limit,
            ),
            prediction=PredictionResponse.from_domain(result.prediction),
            recording_summary=RecordingAnalysisSummaryResponse.from_domain(
                result.recording_summary
            ),
            evidence=EvidenceResponse(
                supporting=tuple(
                    EvidenceItemResponse.from_domain(item)
                    for item in evidence.supporting_evidence
                ),
                conflicting=tuple(
                    EvidenceItemResponse.from_domain(item)
                    for item in evidence.conflicting_evidence
                ),
                neutral=tuple(
                    EvidenceItemResponse.from_domain(item)
                    for item in evidence.neutral_evidence
                ),
                limitations=evidence.limitations,
                evidence_version=evidence.evidence_version,
            ),
            reasoning=ReasoningResponse.from_domain(result.reasoning),
            clinical_report=ClinicalReportResponse.from_domain(
                result.clinical_report
            ),
            narrative=NarrativeResponse.from_domain(result.narrative),
            input_accepted=(
                result.signal_suitability.suitable_for_processing
                if result.signal_suitability is not None
                else True
            ),
            model_prediction_produced=True,
            signal_suitability=(
                SignalSuitabilityResponse(
                    **result.signal_suitability.__dict__
                )
                if result.signal_suitability is not None
                else None
            ),
            ood_assessment=(
                OODAssessmentResponse(**result.ood_assessment.__dict__)
                if result.ood_assessment is not None
                else None
            ),
            analysis_scope=result.analysis_scope,
            model_scope_statement=result.model_scope_statement,
            recommended_interpretation=result.recommended_interpretation,
            analysis_warnings=result.analysis_warnings,
            recording_explanation=(
                RecordingExplanationResponse.from_domain(
                    result.recording_explanation
                )
                if (
                    include_explanations
                    and result.recording_explanation is not None
                )
                else None
            ),
            recording_attribution_overlay=(
                CompactRecordingAttributionOverlayResponse.from_domain(
                    result.recording_attribution_overlay,
                    start_sample=overlay_start_sample,
                    stop_sample=overlay_stop_sample,
                    downsample_limit=overlay_downsample_limit,
                )
                if (
                    include_overlay
                    and result.recording_attribution_overlay is not None
                )
                else None
            ),
        )


class APIErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    error: str
    detail: str
