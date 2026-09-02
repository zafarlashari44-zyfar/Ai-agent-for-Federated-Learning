from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from time import sleep
from types import MethodType

import numpy as np
import pytest
import torch
from torch import nn

from reasoning_pipeline.application.services.explainability_service import (
    ExplainabilityService,
)
from reasoning_pipeline.baseline_adapter.classifier import BaselineClassifier
from reasoning_pipeline.domain.models.beat_analysis_result import (
    BeatAnalysisResult,
)
from reasoning_pipeline.infrastructure.explainability import (
    RecordingAttributionCompositor,
)
from reasoning_pipeline.infrastructure.explainability.grad_cam_1d import (
    GradCAM1D,
)
from reasoning_pipeline.infrastructure.explainability.policies import (
    ExplainAbnormalBeatsPolicy,
)
from reasoning_pipeline.infrastructure.explainability.source_attribution_mapper import (
    SourceAttributionMapper,
)
from reasoning_pipeline.orchestration.model_input_preparer import PreparedBeat

REPO_ROOT = Path(__file__).resolve().parents[4]
CHECKPOINT = (
    REPO_ROOT
    / "fl_ecg_orchestrator"
    / "outputs"
    / "checkpoints"
    / "fedavg_mu_0.0_smote_0_seed_42_final_round_10.pth"
)
CHECKPOINT_HASH = (
    "e9119a15cf9bb8b3d15c085e7be4889"
    "a49f0649ef27682be03a979d6eba0faa2"
)


@pytest.fixture(scope="module")
def classifier() -> BaselineClassifier:
    return BaselineClassifier(
        checkpoint_path=CHECKPOINT,
        device="cpu",
    )


@pytest.fixture
def explainer(classifier: BaselineClassifier) -> GradCAM1D:
    return GradCAM1D(
        model=classifier.model,
        target_layer=classifier.model.features[8],
        target_layer_name="features.8",
    )


def _zero_beat() -> tuple[float, ...]:
    return tuple(float(value) for value in np.zeros(216, dtype=np.float32))


def test_output_shape_metadata_and_finite_values(
    classifier: BaselineClassifier,
    explainer: GradCAM1D,
) -> None:
    beat = _zero_beat()
    prediction = classifier.predict(beat)

    attribution = explainer.explain(
        samples=beat,
        target_class=prediction.predicted_class,
    )

    assert len(attribution.values) == 216
    assert attribution.native_resolution == 54
    assert attribution.method_id == "grad-cam-1d"
    assert attribution.method_version == "1.0.0"
    assert attribution.target_output == "logit"
    assert attribution.interpolation_method == (
        "linear-align-corners-false"
    )
    assert attribution.parameters == (
        ("target_layer", "features.8"),
        ("relu", "true"),
    )
    assert np.all(np.isfinite(attribution.values))
    assert min(attribution.values) >= 0.0
    assert max(attribution.values) <= 1.0


def test_output_is_deterministic(
    classifier: BaselineClassifier,
    explainer: GradCAM1D,
) -> None:
    beat = tuple(
        float(value)
        for value in np.linspace(-1.0, 1.0, 216, dtype=np.float32)
    )
    target_class = classifier.predict(beat).predicted_class

    first = explainer.explain(samples=beat, target_class=target_class)
    second = explainer.explain(samples=beat, target_class=target_class)

    assert first.values == pytest.approx(second.values, abs=1e-8)


def test_hooks_are_removed_after_success_and_failure(
    classifier: BaselineClassifier,
    explainer: GradCAM1D,
) -> None:
    target = classifier.model.features[8]
    forward_hooks_before = len(target._forward_hooks)
    backward_hooks_before = len(target._backward_hooks)

    explainer.explain(samples=_zero_beat(), target_class=2)

    assert len(target._forward_hooks) == forward_hooks_before
    assert len(target._backward_hooks) == backward_hooks_before

    unused_target = nn.Identity()
    failing_explainer = GradCAM1D(
        model=classifier.model,
        target_layer=unused_target,
        target_layer_name="unused",
    )

    with pytest.raises(RuntimeError, match="did not capture"):
        failing_explainer.explain(
            samples=_zero_beat(),
            target_class=2,
        )

    assert not unused_target._forward_hooks
    assert not unused_target._backward_hooks


def test_explanation_does_not_change_prediction_or_weights(
    classifier: BaselineClassifier,
    explainer: GradCAM1D,
) -> None:
    beat = _zero_beat()
    state_before = {
        name: tensor.detach().clone()
        for name, tensor in classifier.model.state_dict().items()
    }
    prediction_before = classifier.predict(beat)

    explainer.explain(
        samples=beat,
        target_class=prediction_before.predicted_class,
    )

    prediction_after = classifier.predict(beat)
    assert prediction_after.predicted_class == prediction_before.predicted_class
    assert prediction_after.probabilities == pytest.approx(
        prediction_before.probabilities,
        abs=1e-8,
    )
    assert all(
        torch.equal(tensor, state_before[name])
        for name, tensor in classifier.model.state_dict().items()
    )
    assert all(
        parameter.grad is None
        for parameter in classifier.model.parameters()
    )


def test_shared_model_lock_serialises_concurrent_explanations(
    classifier: BaselineClassifier,
) -> None:
    first_explainer = GradCAM1D(
        model=classifier.model,
        target_layer=classifier.model.features[8],
        target_layer_name="features.8",
    )
    second_explainer = GradCAM1D(
        model=classifier.model,
        target_layer=classifier.model.features[8],
        target_layer_name="features.8",
    )
    original_forward = classifier.model.forward
    counter_lock = Lock()
    active_calls = 0
    maximum_active_calls = 0

    def tracked_forward(
        _model: nn.Module,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal active_calls, maximum_active_calls
        with counter_lock:
            active_calls += 1
            maximum_active_calls = max(maximum_active_calls, active_calls)
        try:
            sleep(0.02)
            return original_forward(tensor)
        finally:
            with counter_lock:
                active_calls -= 1

    classifier.model.forward = MethodType(  # type: ignore[method-assign]
        tracked_forward,
        classifier.model,
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = (
                executor.submit(
                    current.explain,
                    samples=_zero_beat(),
                    target_class=2,
                )
                for current in (first_explainer, second_explainer)
            )
            results = tuple(future.result() for future in futures)
    finally:
        classifier.model.forward = original_forward  # type: ignore[method-assign]

    assert maximum_active_calls == 1
    assert len(results) == 2


def test_prediction_and_explanation_share_the_same_model_lock(
    classifier: BaselineClassifier,
    explainer: GradCAM1D,
) -> None:
    original_forward = classifier.model.forward
    counter_lock = Lock()
    active_calls = 0
    maximum_active_calls = 0

    def tracked_forward(
        _model: nn.Module,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal active_calls, maximum_active_calls
        with counter_lock:
            active_calls += 1
            maximum_active_calls = max(maximum_active_calls, active_calls)
        try:
            sleep(0.02)
            return original_forward(tensor)
        finally:
            with counter_lock:
                active_calls -= 1

    classifier.model.forward = MethodType(  # type: ignore[method-assign]
        tracked_forward,
        classifier.model,
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            explanation_future = executor.submit(
                explainer.explain,
                samples=_zero_beat(),
                target_class=2,
            )
            prediction_future = executor.submit(
                classifier.predict_many,
                (_zero_beat(), _zero_beat()),
            )
            explanation = explanation_future.result()
            predictions = prediction_future.result()
    finally:
        classifier.model.forward = original_forward  # type: ignore[method-assign]

    assert maximum_active_calls == 1
    assert len(explanation.values) == 216
    assert len(predictions) == 2


def test_attribution_maps_to_source_recording(
    explainer: GradCAM1D,
) -> None:
    start = 1000
    peak = start + 72
    stop = start + 216
    sampling_rate = 360.0
    beat = PreparedBeat(
        beat_index=7,
        r_peak_sample_index=peak,
        source_start_sample_index=start,
        source_stop_sample_index_exclusive=stop,
        r_peak_timestamp_seconds=peak / sampling_rate,
        source_start_timestamp_seconds=start / sampling_rate,
        source_stop_timestamp_seconds_exclusive=stop / sampling_rate,
        sampling_rate_hz=sampling_rate,
        samples=_zero_beat(),
    )
    local = explainer.explain(samples=beat.samples, target_class=2)

    mapped = SourceAttributionMapper().map_to_source(
        prepared_beat=beat,
        attribution=local,
        target_label="V",
    )

    assert mapped.points[0].source_sample_index == 1000
    assert mapped.points[72].source_sample_index == peak
    assert mapped.points[-1].source_sample_index == 1215
    assert mapped.points[72].timestamp_seconds == pytest.approx(peak / 360)


def test_frozen_checkpoint_regression(
    classifier: BaselineClassifier,
    explainer: GradCAM1D,
) -> None:
    beat = _zero_beat()
    prediction = classifier.predict(beat)
    attribution = explainer.explain(
        samples=beat,
        target_class=prediction.predicted_class,
    )

    assert classifier.checkpoint_hash == CHECKPOINT_HASH
    assert prediction.predicted_class == 2
    assert prediction.predicted_label == "V"
    assert prediction.confidence == pytest.approx(
        0.9605029225349426,
        abs=1e-7,
    )
    assert sum(attribution.values) == pytest.approx(
        204.99483324587345,
        abs=1e-5,
    )
    assert max(
        range(216),
        key=lambda index: attribution.values[index],
    ) == 6


def test_real_frozen_model_composes_multiple_segmented_beats(
    classifier: BaselineClassifier,
    explainer: GradCAM1D,
) -> None:
    sampling_rate = 360.0
    prepared_beats = tuple(
        PreparedBeat(
            beat_index=beat_index,
            r_peak_sample_index=start + 72,
            source_start_sample_index=start,
            source_stop_sample_index_exclusive=start + 216,
            r_peak_timestamp_seconds=(start + 72) / sampling_rate,
            source_start_timestamp_seconds=start / sampling_rate,
            source_stop_timestamp_seconds_exclusive=(
                (start + 216) / sampling_rate
            ),
            sampling_rate_hz=sampling_rate,
            samples=samples,
        )
        for beat_index, start, samples in (
            (0, 100, _zero_beat()),
            (1, 300, tuple(1.0 for _ in range(216))),
        )
    )
    predictions = classifier.predict_many(
        tuple(beat.samples for beat in prepared_beats)
    )
    beat_results = tuple(
        BeatAnalysisResult(
            beat_index=beat.beat_index,
            r_peak_sample_index=beat.r_peak_sample_index,
            source_start_sample_index=beat.source_start_sample_index,
            source_stop_sample_index_exclusive=(
                beat.source_stop_sample_index_exclusive
            ),
            r_peak_timestamp_seconds=beat.r_peak_timestamp_seconds,
            source_start_timestamp_seconds=(
                beat.source_start_timestamp_seconds
            ),
            source_stop_timestamp_seconds_exclusive=(
                beat.source_stop_timestamp_seconds_exclusive
            ),
            sampling_rate_hz=beat.sampling_rate_hz,
            prediction=prediction,
        )
        for beat, prediction in zip(
            prepared_beats,
            predictions,
            strict=True,
        )
    )
    service = ExplainabilityService(
        explainers=(explainer,),
        mapper=SourceAttributionMapper(),
        selection_policy=ExplainAbnormalBeatsPolicy(),
    )

    explanation = service.explain_recording(
        record_id="multi-beat-record",
        prepared_beats=prepared_beats,
        beat_results=beat_results,
    )

    assert explanation is not None
    overlay = RecordingAttributionCompositor().compose(
        total_source_samples=700,
        sampling_rate_hz=sampling_rate,
        recording_explanation=explanation,
        method_id="grad-cam-1d",
    )

    assert tuple(prediction.predicted_label for prediction in predictions) == (
        "V",
        "Q",
    )
    assert explanation.total_explained_beats == 2
    assert overlay.explained_beat_count == 2
    assert len(overlay.points) == 700
    assert overlay.points[99].coverage_count == 0
    assert overlay.points[100].coverage_count == 1
    assert overlay.points[315].coverage_count == 2
    assert overlay.points[516].coverage_count == 0
