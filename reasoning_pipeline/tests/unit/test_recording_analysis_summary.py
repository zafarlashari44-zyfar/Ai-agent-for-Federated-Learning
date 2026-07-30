from reasoning_pipeline.domain.models.beat_analysis_result import (
    BeatAnalysisResult,
)
from reasoning_pipeline.domain.models.model_prediction import ModelPrediction
from reasoning_pipeline.domain.models.recording_analysis_summary import (
    RecordingAnalysisSummary,
)


def _beat(beat_index: int, predicted_class: int, label: str) -> BeatAnalysisResult:
    start = beat_index * 300
    peak = start + 72
    sampling_rate = 360.0
    prediction = ModelPrediction(
        predicted_class=predicted_class,
        predicted_label=label,
        probabilities=tuple(
            1.0 if index == predicted_class else 0.0
            for index in range(5)
        ),
        confidence=1.0,
        checkpoint_path="/tmp/model.pth",
        checkpoint_hash="hash",
        model_version="model",
        preprocessing_version="preprocessing",
    )
    return BeatAnalysisResult(
        beat_index=beat_index,
        r_peak_sample_index=peak,
        source_start_sample_index=start,
        source_stop_sample_index_exclusive=start + 216,
        r_peak_timestamp_seconds=peak / sampling_rate,
        source_start_timestamp_seconds=start / sampling_rate,
        source_stop_timestamp_seconds_exclusive=(
            (start + 216) / sampling_rate
        ),
        sampling_rate_hz=sampling_rate,
        prediction=prediction,
    )


def test_recording_summary_aggregates_aami_classes_in_beat_order() -> None:
    beats = (
        _beat(0, 0, "N"),
        _beat(1, 2, "V"),
        _beat(2, 2, "V"),
        _beat(3, 4, "Q"),
    )

    summary = RecordingAnalysisSummary.from_beat_results(beats)

    assert summary.total_valid_beats == 4
    assert summary.class_counts == (
        ("N", 1),
        ("S", 0),
        ("V", 2),
        ("F", 0),
        ("Q", 1),
    )
    assert summary.abnormal_beat_count == 3
    assert summary.abnormal_beat_percentage == 75.0
    assert summary.dominant_predicted_class == 2
    assert summary.dominant_predicted_label == "V"
    assert summary.beat_results is beats
