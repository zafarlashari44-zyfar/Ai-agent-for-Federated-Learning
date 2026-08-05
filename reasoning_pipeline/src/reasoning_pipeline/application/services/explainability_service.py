from __future__ import annotations

from typing import TYPE_CHECKING

from reasoning_pipeline.application.ports.attribution_mapper import (
    AttributionMapperProtocol,
)
from reasoning_pipeline.application.ports.beat_explainer import (
    BeatExplainerProtocol,
)
from reasoning_pipeline.application.ports.explanation_selection_policy import (
    ExplanationSelectionPolicyProtocol,
)
from reasoning_pipeline.domain.models.beat_analysis_result import (
    BeatAnalysisResult,
)
from reasoning_pipeline.domain.models.beat_explanation import BeatExplanation
from reasoning_pipeline.domain.models.recording_explanation import (
    RecordingExplanation,
)

if TYPE_CHECKING:
    from reasoning_pipeline.orchestration.model_input_preparer import PreparedBeat


class ExplainabilityService:
    """Coordinate independently injected beat explainers and source mapping."""

    def __init__(
        self,
        *,
        explainers: tuple[BeatExplainerProtocol, ...],
        mapper: AttributionMapperProtocol,
        selection_policy: ExplanationSelectionPolicyProtocol,
    ) -> None:
        method_ids = tuple(explainer.method_id for explainer in explainers)
        if len(set(method_ids)) != len(method_ids):
            raise ValueError("Explainer method IDs must be unique")

        self._explainers = explainers
        self._mapper = mapper
        self._selection_policy = selection_policy

    def explain_recording(
        self,
        *,
        record_id: str,
        prepared_beats: tuple[PreparedBeat, ...],
        beat_results: tuple[BeatAnalysisResult, ...],
    ) -> RecordingExplanation | None:
        if len(prepared_beats) != len(beat_results):
            raise ValueError(
                "Prepared beats and beat results must have equal lengths"
            )
        if not beat_results or not self._explainers:
            return None

        prepared_by_index = {
            beat.beat_index: beat
            for beat in prepared_beats
        }
        result_by_index = {
            result.beat_index: result
            for result in beat_results
        }

        if len(prepared_by_index) != len(prepared_beats):
            raise ValueError("Prepared beat indices must be unique")
        if len(result_by_index) != len(beat_results):
            raise ValueError("Beat result indices must be unique")
        if prepared_by_index.keys() != result_by_index.keys():
            raise ValueError(
                "Prepared beats and beat results must share beat indices"
            )

        selected_indices = self._selection_policy.select(beat_results)
        if len(set(selected_indices)) != len(selected_indices):
            raise ValueError("Selected beat indices must be unique")

        beat_explanations: list[BeatExplanation] = []
        completed_methods: list[str] = []

        for beat_index in selected_indices:
            try:
                prepared_beat = prepared_by_index[beat_index]
                beat_result = result_by_index[beat_index]
            except KeyError as exc:
                raise ValueError(
                    f"Selection policy returned unknown beat index {beat_index}"
                ) from exc

            prediction = beat_result.prediction
            attribution_maps = []

            for explainer in self._explainers:
                local_attribution = explainer.explain(
                    samples=prepared_beat.samples,
                    target_class=prediction.predicted_class,
                )
                if local_attribution is None:
                    continue

                attribution_maps.append(
                    self._mapper.map_to_source(
                        prepared_beat=prepared_beat,
                        attribution=local_attribution,
                        target_label=prediction.predicted_label,
                    )
                )
                if explainer.method_id not in completed_methods:
                    completed_methods.append(explainer.method_id)

            if not attribution_maps:
                continue

            beat_explanations.append(
                BeatExplanation(
                    beat_index=prepared_beat.beat_index,
                    r_peak_sample_index=prepared_beat.r_peak_sample_index,
                    r_peak_timestamp_seconds=(
                        prepared_beat.r_peak_timestamp_seconds
                    ),
                    source_start_sample_index=(
                        prepared_beat.source_start_sample_index
                    ),
                    source_stop_sample_index_exclusive=(
                        prepared_beat.source_stop_sample_index_exclusive
                    ),
                    source_start_timestamp_seconds=(
                        prepared_beat.source_start_timestamp_seconds
                    ),
                    source_stop_timestamp_seconds_exclusive=(
                        prepared_beat.source_stop_timestamp_seconds_exclusive
                    ),
                    sampling_rate_hz=prepared_beat.sampling_rate_hz,
                    predicted_class=prediction.predicted_class,
                    predicted_label=prediction.predicted_label,
                    prediction_confidence=prediction.confidence,
                    attribution_maps=tuple(attribution_maps),
                )
            )

        if not beat_explanations:
            return None

        first_prediction = beat_results[0].prediction
        return RecordingExplanation(
            record_id=record_id,
            selection_policy=self._selection_policy.policy_id,
            total_valid_beats=len(beat_results),
            total_explained_beats=len(beat_explanations),
            beat_explanations=tuple(beat_explanations),
            requested_methods=tuple(
                explainer.method_id
                for explainer in self._explainers
            ),
            completed_methods=tuple(completed_methods),
            model_version=first_prediction.model_version,
            checkpoint_hash=first_prediction.checkpoint_hash,
            preprocessing_version=first_prediction.preprocessing_version,
        )
