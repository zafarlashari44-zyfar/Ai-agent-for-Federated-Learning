import pytest

from reasoning_pipeline.api.schemas.analyse import (
    CompactRecordingAttributionOverlayResponse,
    RecordingAttributionOverlayResponse,
    RecordingExplanationResponse,
)
from reasoning_pipeline.domain.models.attribution_map import AttributionMap
from reasoning_pipeline.domain.models.attribution_point import AttributionPoint
from reasoning_pipeline.domain.models.beat_explanation import BeatExplanation
from reasoning_pipeline.domain.models.recording_explanation import (
    RecordingExplanation,
)
from reasoning_pipeline.infrastructure.explainability import (
    RecordingAttributionCompositor,
)

SAMPLING_RATE = 360.0
METHOD_ID = "grad-cam-1d"
METHOD_VERSION = "1.0.0"


def _attribution_map(
    *,
    start: int,
    value: float,
) -> AttributionMap:
    return AttributionMap(
        method_id=METHOD_ID,
        method_version=METHOD_VERSION,
        target_class=2,
        target_label="V",
        target_output="logit",
        points=tuple(
            AttributionPoint(
                beat_sample_index=local_index,
                source_sample_index=start + local_index,
                timestamp_seconds=(start + local_index) / SAMPLING_RATE,
                attribution=value,
                input_value=0.0,
            )
            for local_index in range(216)
        ),
        signed=False,
        native_resolution=54,
        interpolation_method="linear-align-corners-false",
        normalisation="relu-min-max",
        sampling_rate_hz=SAMPLING_RATE,
        source_start_sample_index=start,
        source_stop_sample_index_exclusive=start + 216,
    )


def _beat_explanation(
    *,
    beat_index: int,
    start: int,
    value: float,
) -> BeatExplanation:
    peak = start + 72
    stop = start + 216
    return BeatExplanation(
        beat_index=beat_index,
        r_peak_sample_index=peak,
        r_peak_timestamp_seconds=peak / SAMPLING_RATE,
        source_start_sample_index=start,
        source_stop_sample_index_exclusive=stop,
        source_start_timestamp_seconds=start / SAMPLING_RATE,
        source_stop_timestamp_seconds_exclusive=stop / SAMPLING_RATE,
        sampling_rate_hz=SAMPLING_RATE,
        predicted_class=2,
        predicted_label="V",
        prediction_confidence=0.9,
        attribution_maps=(
            _attribution_map(start=start, value=value),
        ),
    )


def _recording_explanation(
    *beats: BeatExplanation,
) -> RecordingExplanation:
    return RecordingExplanation(
        record_id="record-001",
        selection_policy="explain-abnormal-beats",
        total_valid_beats=len(beats),
        total_explained_beats=len(beats),
        beat_explanations=beats,
        requested_methods=(METHOD_ID,),
        completed_methods=(METHOD_ID,),
        model_version="model",
        checkpoint_hash="hash",
        preprocessing_version="preprocessing",
    )


def test_non_overlapping_windows_and_uncovered_samples() -> None:
    explanation = _recording_explanation(
        _beat_explanation(beat_index=1, start=10, value=0.25),
        _beat_explanation(beat_index=2, start=250, value=0.75),
    )

    overlay = RecordingAttributionCompositor().compose(
        total_source_samples=500,
        sampling_rate_hz=SAMPLING_RATE,
        recording_explanation=explanation,
        method_id=METHOD_ID,
    )

    assert len(overlay.points) == 500
    assert overlay.points[0].maximum_attribution == 0.0
    assert overlay.points[0].mean_attribution == 0.0
    assert overlay.points[0].coverage_count == 0
    assert overlay.points[0].contributing_beat_indices == ()
    assert overlay.points[10].maximum_attribution == 0.25
    assert overlay.points[10].coverage_count == 1
    assert overlay.points[250].maximum_attribution == 0.75
    assert overlay.explained_beat_count == 2


def test_overlapping_windows_calculate_maximum_mean_and_coverage() -> None:
    explanation = _recording_explanation(
        _beat_explanation(beat_index=3, start=100, value=0.2),
        _beat_explanation(beat_index=8, start=200, value=0.8),
    )

    overlay = RecordingAttributionCompositor().compose(
        total_source_samples=500,
        sampling_rate_hz=SAMPLING_RATE,
        recording_explanation=explanation,
        method_id=METHOD_ID,
    )

    overlap = overlay.points[200]
    assert overlay.aggregation_method == "maximum"
    assert overlap.maximum_attribution == pytest.approx(0.8)
    assert overlap.mean_attribution == pytest.approx(0.5)
    assert overlap.coverage_count == 2
    assert overlap.contributing_beat_indices == (3, 8)
    assert overlay.points[199].coverage_count == 1
    assert overlay.points[315].coverage_count == 2
    assert overlay.points[316].coverage_count == 1


def test_output_indices_timestamps_and_contributors_are_deterministic() -> None:
    explanation = _recording_explanation(
        _beat_explanation(beat_index=3, start=0, value=0.3),
        _beat_explanation(beat_index=9, start=50, value=0.6),
    )
    compositor = RecordingAttributionCompositor()

    first = compositor.compose(
        total_source_samples=300,
        sampling_rate_hz=SAMPLING_RATE,
        recording_explanation=explanation,
        method_id=METHOD_ID,
    )
    second = compositor.compose(
        total_source_samples=300,
        sampling_rate_hz=SAMPLING_RATE,
        recording_explanation=explanation,
        method_id=METHOD_ID,
    )

    assert first == second
    assert tuple(point.source_sample_index for point in first.points) == tuple(
        range(300)
    )
    assert first.points[123].timestamp_seconds == pytest.approx(123 / 360)
    assert first.points[123].contributing_beat_indices == (3, 9)


def test_mapped_point_outside_recording_is_rejected() -> None:
    explanation = _recording_explanation(
        _beat_explanation(beat_index=1, start=400, value=0.5),
    )

    with pytest.raises(ValueError, match="exceeds recording bounds"):
        RecordingAttributionCompositor().compose(
            total_source_samples=500,
            sampling_rate_hz=SAMPLING_RATE,
            recording_explanation=explanation,
            method_id=METHOD_ID,
        )


def test_explanation_and_overlay_serialize_as_consistent_json_arrays() -> None:
    explanation = _recording_explanation(
        _beat_explanation(beat_index=1, start=10, value=0.5),
    )
    overlay = RecordingAttributionCompositor().compose(
        total_source_samples=250,
        sampling_rate_hz=SAMPLING_RATE,
        recording_explanation=explanation,
        method_id=METHOD_ID,
    )

    explanation_json = RecordingExplanationResponse.from_domain(
        explanation
    ).model_dump(mode="json")
    overlay_json = RecordingAttributionOverlayResponse.from_domain(
        overlay
    ).model_dump(mode="json")

    assert isinstance(explanation_json["beat_explanations"], list)
    attribution_maps = explanation_json["beat_explanations"][0][
        "attribution_maps"
    ]
    assert isinstance(attribution_maps[0]["points"], list)
    assert len(attribution_maps[0]["points"]) == 216
    assert isinstance(overlay_json["points"], list)
    assert len(overlay_json["points"]) == 250
    assert overlay_json["points"][10]["contributing_beat_indices"] == [1]


def test_compact_overlay_parallel_arrays_preserve_exact_alignment() -> None:
    explanation = _recording_explanation(
        _beat_explanation(beat_index=1, start=10, value=0.5),
    )
    overlay = RecordingAttributionCompositor().compose(
        total_source_samples=250,
        sampling_rate_hz=SAMPLING_RATE,
        recording_explanation=explanation,
        method_id=METHOD_ID,
    )

    compact = CompactRecordingAttributionOverlayResponse.from_domain(overlay)
    payload = compact.model_dump(mode="json")
    array_lengths = {
        len(payload[field])
        for field in (
            "sample_indices",
            "timestamps_seconds",
            "maximum_attributions",
            "mean_attributions",
            "coverage_counts",
            "contributing_beat_indices",
        )
    }

    assert array_lengths == {250}
    assert payload["sample_indices"] == list(range(250))
    assert payload["timestamps_seconds"][37] == pytest.approx(37 / 360)
    assert payload["maximum_attributions"][10] == 0.5
    assert payload["coverage_counts"][10] == 1
    assert payload["contributing_beat_indices"][10] == [1]


def test_compact_overlay_window_retains_original_source_coordinates() -> None:
    explanation = _recording_explanation(
        _beat_explanation(beat_index=2, start=10, value=0.4),
    )
    overlay = RecordingAttributionCompositor().compose(
        total_source_samples=300,
        sampling_rate_hz=SAMPLING_RATE,
        recording_explanation=explanation,
        method_id=METHOD_ID,
    )

    compact = CompactRecordingAttributionOverlayResponse.from_domain(
        overlay,
        start_sample=100,
        stop_sample=180,
    )

    assert compact.start_sample_index == 100
    assert compact.stop_sample_index_exclusive == 180
    assert compact.source_window_sample_count == 80
    assert compact.sample_indices == tuple(range(100, 180))
    assert compact.timestamps_seconds[0] == pytest.approx(100 / 360)


@pytest.mark.parametrize(
    ("start", "stop", "message"),
    (
        (-1, 10, "must be non-negative"),
        (10, 301, "cannot exceed"),
        (20, 20, "must be less than"),
        (21, 20, "must be less than"),
    ),
)
def test_compact_overlay_rejects_invalid_recording_windows(
    start: int,
    stop: int,
    message: str,
) -> None:
    explanation = _recording_explanation(
        _beat_explanation(beat_index=2, start=10, value=0.4),
    )
    overlay = RecordingAttributionCompositor().compose(
        total_source_samples=300,
        sampling_rate_hz=SAMPLING_RATE,
        recording_explanation=explanation,
        method_id=METHOD_ID,
    )

    with pytest.raises(ValueError, match=message):
        CompactRecordingAttributionOverlayResponse.from_domain(
            overlay,
            start_sample=start,
            stop_sample=stop,
        )


def test_downsampling_retains_each_bin_maximum_and_exact_alignment() -> None:
    explanation = _recording_explanation(
        _beat_explanation(beat_index=2, start=10, value=0.4),
    )
    overlay = RecordingAttributionCompositor().compose(
        total_source_samples=300,
        sampling_rate_hz=SAMPLING_RATE,
        recording_explanation=explanation,
        method_id=METHOD_ID,
    )
    points = list(overlay.points)
    points[49] = points[49].__class__(
        source_sample_index=49,
        timestamp_seconds=49 / SAMPLING_RATE,
        maximum_attribution=0.91,
        mean_attribution=0.91,
        coverage_count=1,
        contributing_beat_indices=(2,),
    )
    points[199] = points[199].__class__(
        source_sample_index=199,
        timestamp_seconds=199 / SAMPLING_RATE,
        maximum_attribution=1.0,
        mean_attribution=1.0,
        coverage_count=1,
        contributing_beat_indices=(2,),
    )
    overlay = overlay.__class__(
        record_id=overlay.record_id,
        method_id=overlay.method_id,
        method_version=overlay.method_version,
        sampling_rate_hz=overlay.sampling_rate_hz,
        total_source_samples=overlay.total_source_samples,
        aggregation_method=overlay.aggregation_method,
        points=tuple(points),
        explained_beat_count=overlay.explained_beat_count,
        warnings=overlay.warnings,
    )

    compact = CompactRecordingAttributionOverlayResponse.from_domain(
        overlay,
        downsample_limit=3,
    )

    assert compact.downsampled
    assert compact.downsampling_strategy == (
        "contiguous-bin-maximum-attribution"
    )
    assert compact.sample_indices == (49, 199, 200)
    assert compact.maximum_attributions == (0.91, 1.0, 0.4)
    assert all(
        timestamp == pytest.approx(index / SAMPLING_RATE)
        for index, timestamp in zip(
            compact.sample_indices,
            compact.timestamps_seconds,
            strict=True,
        )
    )
