from dataclasses import FrozenInstanceError

import pytest

from reasoning_pipeline.application.ports.beat_explainer import (
    LocalAttribution,
)
from reasoning_pipeline.application.services.explainability_service import (
    ExplainabilityService,
)
from reasoning_pipeline.domain.models.beat_analysis_result import (
    BeatAnalysisResult,
)
from reasoning_pipeline.domain.models.model_prediction import ModelPrediction
from reasoning_pipeline.infrastructure.explainability.noop import (
    NoOpBeatExplainer,
)
from reasoning_pipeline.infrastructure.explainability.policies import (
    ExplainAbnormalBeatsPolicy,
)
from reasoning_pipeline.infrastructure.explainability.source_attribution_mapper import (
    SourceAttributionMapper,
)
from reasoning_pipeline.orchestration.model_input_preparer import PreparedBeat


def _prepared_beat(beat_index: int = 2) -> PreparedBeat:
    start = 328
    stop = start + 216
    peak = start + 72
    sampling_rate = 360.0
    return PreparedBeat(
        beat_index=beat_index,
        r_peak_sample_index=peak,
        source_start_sample_index=start,
        source_stop_sample_index_exclusive=stop,
        r_peak_timestamp_seconds=peak / sampling_rate,
        source_start_timestamp_seconds=start / sampling_rate,
        source_stop_timestamp_seconds_exclusive=stop / sampling_rate,
        sampling_rate_hz=sampling_rate,
        samples=tuple(float(index) for index in range(216)),
    )


def _prediction(predicted_class: int, label: str) -> ModelPrediction:
    return ModelPrediction(
        predicted_class=predicted_class,
        predicted_label=label,
        probabilities=tuple(
            1.0 if index == predicted_class else 0.0
            for index in range(5)
        ),
        confidence=1.0,
        checkpoint_path="/tmp/model.pth",
        checkpoint_hash="checkpoint-hash",
        model_version="model-version",
        preprocessing_version="preprocessing-version",
    )


def _beat_result(
    beat: PreparedBeat,
    prediction: ModelPrediction,
) -> BeatAnalysisResult:
    return BeatAnalysisResult(
        beat_index=beat.beat_index,
        r_peak_sample_index=beat.r_peak_sample_index,
        source_start_sample_index=beat.source_start_sample_index,
        source_stop_sample_index_exclusive=(
            beat.source_stop_sample_index_exclusive
        ),
        r_peak_timestamp_seconds=beat.r_peak_timestamp_seconds,
        source_start_timestamp_seconds=beat.source_start_timestamp_seconds,
        source_stop_timestamp_seconds_exclusive=(
            beat.source_stop_timestamp_seconds_exclusive
        ),
        sampling_rate_hz=beat.sampling_rate_hz,
        prediction=prediction,
    )


class FakeBeatExplainer:
    @property
    def method_id(self) -> str:
        return "fake"

    def explain(
        self,
        *,
        samples: tuple[float, ...],
        target_class: int,
    ) -> LocalAttribution:
        return LocalAttribution(
            method_id=self.method_id,
            method_version="1.0",
            target_class=target_class,
            target_output="logit",
            values=tuple(float(index) / 215 for index in range(216)),
            signed=False,
            native_resolution=216,
            interpolation_method=None,
            normalisation="none",
        )


def test_source_mapper_maps_all_points_to_original_recording() -> None:
    beat = _prepared_beat()
    local = FakeBeatExplainer().explain(
        samples=beat.samples,
        target_class=2,
    )

    mapped = SourceAttributionMapper().map_to_source(
        prepared_beat=beat,
        attribution=local,
        target_label="V",
    )

    assert len(mapped.points) == 216
    assert mapped.points[0].beat_sample_index == 0
    assert mapped.points[0].source_sample_index == 328
    assert mapped.points[0].timestamp_seconds == pytest.approx(328 / 360)
    assert mapped.points[-1].beat_sample_index == 215
    assert mapped.points[-1].source_sample_index == 543
    assert mapped.points[-1].timestamp_seconds == pytest.approx(543 / 360)
    assert mapped.points[72].source_sample_index == beat.r_peak_sample_index


def test_explain_abnormal_policy_preserves_original_beat_indices() -> None:
    normal_beat = _prepared_beat(2)
    abnormal_beat = _prepared_beat(5)
    results = (
        _beat_result(normal_beat, _prediction(0, "N")),
        _beat_result(abnormal_beat, _prediction(2, "V")),
    )

    selected = ExplainAbnormalBeatsPolicy().select(results)

    assert selected == (5,)


def test_service_builds_recording_explanation_from_injected_components() -> None:
    beat = _prepared_beat()
    result = _beat_result(beat, _prediction(2, "V"))
    service = ExplainabilityService(
        explainers=(FakeBeatExplainer(),),
        mapper=SourceAttributionMapper(),
        selection_policy=ExplainAbnormalBeatsPolicy(),
    )

    explanation = service.explain_recording(
        record_id="record-001",
        prepared_beats=(beat,),
        beat_results=(result,),
    )

    assert explanation is not None
    assert explanation.record_id == "record-001"
    assert explanation.total_valid_beats == 1
    assert explanation.total_explained_beats == 1
    assert explanation.requested_methods == ("fake",)
    assert explanation.completed_methods == ("fake",)
    assert explanation.beat_explanations[0].beat_index == 2
    assert explanation.beat_explanations[0].attribution_maps[0].method_id == (
        "fake"
    )

    with pytest.raises(FrozenInstanceError):
        explanation.total_valid_beats = 2  # type: ignore[misc]


def test_noop_explainer_produces_no_recording_explanation() -> None:
    beat = _prepared_beat()
    result = _beat_result(beat, _prediction(2, "V"))
    service = ExplainabilityService(
        explainers=(NoOpBeatExplainer(),),
        mapper=SourceAttributionMapper(),
        selection_policy=ExplainAbnormalBeatsPolicy(),
    )

    explanation = service.explain_recording(
        record_id="record-001",
        prepared_beats=(beat,),
        beat_results=(result,),
    )

    assert explanation is None
