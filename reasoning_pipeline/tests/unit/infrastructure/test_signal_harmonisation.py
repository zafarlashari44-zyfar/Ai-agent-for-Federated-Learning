from pathlib import Path
from typing import cast

import numpy as np
import pytest

from reasoning_pipeline.application.services.pipeline_service import (
    PipelineService,
)
from reasoning_pipeline.domain.enums.statuses import SignalSuitabilityStatus
from reasoning_pipeline.domain.models.ecg_signal import ECGSignal
from reasoning_pipeline.domain.models.signal_suitability_assessment import (
    SignalSuitabilityAssessment,
)
from reasoning_pipeline.infrastructure.signal_harmonisation import (
    ScipySignalHarmoniser,
)
from reasoning_pipeline.orchestration.analysis_result import (
    ECGAnalysisResult,
)
from reasoning_pipeline.orchestration.model_input_preparer import (
    ModelInputPreparer,
)

TARGET_RATE = ModelInputPreparer.EXPECTED_SAMPLING_RATE_HZ


def make_signal(
    *,
    sampling_rate_hz: float = 360.0,
    units: str | None = "mV",
    seconds: float = 4.0,
    source_format: str = "csv",
) -> ECGSignal:
    count = int(sampling_rate_hz * seconds)
    samples = tuple(
        float(value)
        for value in np.sin(np.linspace(0.0, seconds * 8.0, count))
    )
    return ECGSignal(
        record_id="record-001",
        samples=samples,
        sampling_rate_hz=sampling_rate_hz,
        source="test-upload",
        lead_name="II",
        source_format=source_format,
        original_sampling_rate_hz=sampling_rate_hz,
        lead_names=("I", "II"),
        units=units,
        original_sample_count=count,
        original_duration_seconds=seconds,
        warnings=("source warning",),
    )


def harmoniser() -> ScipySignalHarmoniser:
    return ScipySignalHarmoniser(target_sampling_rate_hz=TARGET_RATE)


def test_360_hz_input_is_not_resampled() -> None:
    result = harmoniser().harmonise(make_signal())
    assert result.sampling_rate_hz == 360.0
    assert not result.resampled
    assert result.resampling_method is None
    assert result.resampling_up_factor == 1
    assert result.resampling_down_factor == 1


@pytest.mark.parametrize(
    ("source_rate", "up", "down"),
    [(250.0, 36, 25), (500.0, 18, 25), (100.0, 18, 5)],
)
def test_resamples_common_ecg_rates(
    source_rate: float,
    up: int,
    down: int,
) -> None:
    result = harmoniser().harmonise(make_signal(sampling_rate_hz=source_rate))
    assert result.sampling_rate_hz == TARGET_RATE
    assert result.resampled
    assert result.resampling_method == "scipy.signal.resample_poly"
    assert result.resampling_up_factor == up
    assert result.resampling_down_factor == down
    assert result.sample_count == 4 * 360
    assert all(np.isfinite(result.samples))


@pytest.mark.parametrize(
    ("units", "value", "expected"),
    [("V", 1.0, 1000.0), ("uV", 1000.0, 1.0), ("µV", 1000.0, 1.0), ("mV", 1.0, 1.0)],
)
def test_unit_conversion(units: str, value: float, expected: float) -> None:
    signal = make_signal(units=units)
    signal = ECGSignal(
        **{
            **signal.__dict__,
            "samples": (value,) * signal.sample_count,
        }
    )
    result = harmoniser().harmonise(signal)
    assert result.samples[0] == pytest.approx(expected)
    assert result.original_units == units
    assert result.units == "mV"
    assert result.target_units == "mV"


def test_unsupported_units_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported or ambiguous"):
        harmoniser().harmonise(make_signal(units="counts"))


def test_missing_units_are_rejected_for_non_npy() -> None:
    with pytest.raises(ValueError, match="amplitude units are required"):
        harmoniser().harmonise(make_signal(units=None))


def test_legacy_npy_missing_units_uses_compatibility_warning() -> None:
    result = harmoniser().harmonise(
        make_signal(units=None, source_format="npy")
    )
    assert result.units == "mV"
    assert result.harmonisation_warnings == (
        "Legacy NPY compatibility mode used: missing amplitude units were "
        "treated as mV.",
    )


@pytest.mark.parametrize("rate", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_sampling_rates_are_rejected(rate: float) -> None:
    signal = make_signal()
    object.__setattr__(signal, "sampling_rate_hz", rate)
    with pytest.raises(ValueError, match="finite and positive"):
        harmoniser().harmonise(signal)


def test_non_integral_resampling_rate_is_rejected() -> None:
    signal = make_signal()
    object.__setattr__(signal, "sampling_rate_hz", 250.5)
    with pytest.raises(ValueError, match="must be integral"):
        harmoniser().harmonise(signal)


def test_duration_and_source_metadata_are_preserved() -> None:
    service = harmoniser()
    source = make_signal(sampling_rate_hz=250.0)
    result = service.harmonise(source)
    assert abs(result.duration_seconds - source.duration_seconds) <= (
        service.duration_tolerance_seconds
    )
    assert result.record_id == source.record_id
    assert result.source_format == source.source_format
    assert result.original_sampling_rate_hz == 250.0
    assert result.original_sample_count == source.sample_count
    assert result.original_duration_seconds == source.duration_seconds
    assert result.lead_names == source.lead_names
    assert result.lead_name == source.lead_name
    assert result.warnings == source.warnings


class Adapter:
    supported_suffixes = (".csv",)

    def __init__(self, signal: ECGSignal) -> None:
        self.signal = signal

    def supports(self, file_path: str | Path) -> bool:
        return Path(file_path).suffix == ".csv"

    def load(self, **kwargs: object) -> ECGSignal:
        return self.signal


class Pipeline:
    def __init__(self) -> None:
        self.received_signal: ECGSignal | None = None

    def analyse(
        self,
        signal: ECGSignal,
        *,
        suitability_assessment: SignalSuitabilityAssessment | None = None,
    ) -> ECGAnalysisResult:
        self.received_signal = signal
        return cast(ECGAnalysisResult, object())


class SuitabilityAssessor:
    def assess(self, signal: ECGSignal) -> SignalSuitabilityAssessment:
        return SignalSuitabilityAssessment(
            status=SignalSuitabilityStatus.ACCEPTED,
            suitable_for_processing=True,
            quality_score=1.0,
            duration_seconds=signal.duration_seconds,
            sampling_rate_hz=signal.sampling_rate_hz,
            selected_lead=signal.lead_name,
            units=signal.units,
            detected_r_peak_count=1,
            estimated_heart_rate_bpm=None,
            finite_sample_ratio=1.0,
            flatline_percentage=0.0,
            clipping_percentage=0.0,
            noise_score=0.0,
        )


def test_adapter_harmoniser_pipeline_integration() -> None:
    pipeline = Pipeline()
    service = PipelineService(
        pipeline=pipeline,
        input_adapters=(Adapter(make_signal(sampling_rate_hz=250.0)),),
        signal_harmoniser=harmoniser(),
        suitability_assessor=SuitabilityAssessor(),
    )
    result = service.analyse_file(file_path="record.csv")
    assert result is not None
    assert pipeline.received_signal is not None
    assert pipeline.received_signal.sampling_rate_hz == 360.0
    assert pipeline.received_signal.resampled
