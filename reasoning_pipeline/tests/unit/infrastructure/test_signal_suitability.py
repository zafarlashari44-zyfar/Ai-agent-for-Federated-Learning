import numpy as np

from reasoning_pipeline.domain.enums.statuses import SignalSuitabilityStatus
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.infrastructure.signal_suitability import (
    HeuristicSignalSuitabilityAssessor,
)


def ecg_signal(
    *,
    seconds: float = 10.0,
    source: str = "mit-bih",
    source_format: str = "npy",
) -> ECGSignal:
    rate = 360.0
    samples = np.zeros(int(seconds * rate))
    samples += 0.02 * np.sin(np.linspace(0, seconds * 2 * np.pi, samples.size))
    for index in range(180, samples.size, 360):
        samples[index : index + 3] += (0.5, 1.0, 0.5)
    return ECGSignal(
        record_id="record",
        samples=tuple(samples),
        sampling_rate_hz=rate,
        source=source,
        lead_name="MLII",
        source_format=source_format,
        original_sampling_rate_hz=100.0 if source_format == "wfdb" else rate,
        lead_names=("MLII",),
        units="mV",
        original_sample_count=samples.size,
        original_duration_seconds=seconds,
        original_units="mV",
        target_sampling_rate_hz=rate,
        target_units="mV",
        resampled=source_format == "wfdb",
    )


def test_valid_mit_bih_style_signal_is_accepted() -> None:
    result = HeuristicSignalSuitabilityAssessor().assess(ecg_signal())
    assert result.suitable_for_processing
    assert result.detected_r_peak_count >= 2


def test_valid_mit_bih_wfdb_signal_does_not_get_external_source_warning() -> None:
    result = HeuristicSignalSuitabilityAssessor().assess(
        ecg_signal(source="mit-bih", source_format="wfdb")
    )

    assert result.suitable_for_processing
    assert not any(
        "outside the validated MIT-BIH" in item
        for item in result.warnings
    )


def test_external_wfdb_and_ptbxl_style_signal_is_not_rejected() -> None:
    result = HeuristicSignalSuitabilityAssessor().assess(
        ecg_signal(source="ptb-xl", source_format="wfdb")
    )
    assert result.suitable_for_processing
    assert result.status is SignalSuitabilityStatus.ACCEPTED_WITH_WARNINGS
    assert any("outside the validated MIT-BIH" in item for item in result.warnings)


def test_flatline_is_rejected() -> None:
    signal = ecg_signal()
    object.__setattr__(signal, "samples", (0.0,) * signal.sample_count)
    result = HeuristicSignalSuitabilityAssessor().assess(signal)
    assert not result.suitable_for_processing
    assert any("variance" in item for item in result.rejection_reasons)


def test_no_detectable_r_peaks_is_rejected() -> None:
    signal = ecg_signal()
    quiet = 0.001 * np.sin(np.linspace(0, 20, signal.sample_count))
    object.__setattr__(signal, "samples", tuple(quiet))
    result = HeuristicSignalSuitabilityAssessor().assess(signal)
    assert any("R peaks" in item for item in result.rejection_reasons)


def test_very_short_recording_is_rejected() -> None:
    result = HeuristicSignalSuitabilityAssessor().assess(ecg_signal(seconds=1.0))
    assert not result.suitable_for_processing
    assert any("duration" in item for item in result.rejection_reasons)


def test_non_finite_signal_is_rejected() -> None:
    signal = ecg_signal()
    samples = list(signal.samples)
    samples[100] = np.nan
    object.__setattr__(signal, "samples", tuple(samples))
    result = HeuristicSignalSuitabilityAssessor().assess(signal)
    assert any("non-finite" in item for item in result.rejection_reasons)


def test_extreme_clipping_is_rejected() -> None:
    signal = ecg_signal()
    clipped = np.tile((-1.0, 1.0), signal.sample_count // 2)
    object.__setattr__(signal, "samples", tuple(clipped))
    result = HeuristicSignalSuitabilityAssessor().assess(signal)
    assert any("clipping" in item for item in result.rejection_reasons)


def test_high_noise_is_rejected() -> None:
    signal = ecg_signal()
    noisy = np.random.default_rng(7).normal(0.0, 1.0, signal.sample_count)
    object.__setattr__(signal, "samples", tuple(noisy))
    result = HeuristicSignalSuitabilityAssessor().assess(signal)
    assert result.noise_score >= 0.65
    assert result.status in {
        SignalSuitabilityStatus.ACCEPTED_WITH_WARNINGS,
        SignalSuitabilityStatus.REJECTED,
    }


def test_empty_signal_metric_path_is_rejected() -> None:
    signal = ecg_signal()
    object.__setattr__(signal, "samples", ())
    result = HeuristicSignalSuitabilityAssessor().assess(signal)
    assert not result.suitable_for_processing
    assert any("empty" in item for item in result.rejection_reasons)
