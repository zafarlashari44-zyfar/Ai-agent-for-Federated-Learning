from __future__ import annotations

from typing import TYPE_CHECKING

from reasoning_pipeline.application.ports.beat_explainer import (
    LocalAttribution,
)
from reasoning_pipeline.domain.models.attribution_map import AttributionMap
from reasoning_pipeline.domain.models.attribution_point import AttributionPoint

if TYPE_CHECKING:
    from reasoning_pipeline.orchestration.model_input_preparer import PreparedBeat


class SourceAttributionMapper:
    """Map prepared-beat attribution positions to source ECG coordinates."""

    def map_to_source(
        self,
        *,
        prepared_beat: PreparedBeat,
        attribution: LocalAttribution,
        target_label: str,
    ) -> AttributionMap:
        if len(attribution.values) != len(prepared_beat.samples):
            raise ValueError(
                "Attribution values must match prepared beat samples"
            )

        points = tuple(
            AttributionPoint(
                beat_sample_index=local_index,
                source_sample_index=(
                    prepared_beat.source_start_sample_index + local_index
                ),
                timestamp_seconds=(
                    (
                        prepared_beat.source_start_sample_index
                        + local_index
                    )
                    / prepared_beat.sampling_rate_hz
                ),
                attribution=value,
                input_value=prepared_beat.samples[local_index],
            )
            for local_index, value in enumerate(attribution.values)
        )

        return AttributionMap(
            method_id=attribution.method_id,
            method_version=attribution.method_version,
            target_class=attribution.target_class,
            target_label=target_label,
            target_output=attribution.target_output,
            points=points,
            signed=attribution.signed,
            native_resolution=attribution.native_resolution,
            interpolation_method=attribution.interpolation_method,
            normalisation=attribution.normalisation,
            sampling_rate_hz=prepared_beat.sampling_rate_hz,
            source_start_sample_index=(
                prepared_beat.source_start_sample_index
            ),
            source_stop_sample_index_exclusive=(
                prepared_beat.source_stop_sample_index_exclusive
            ),
            convergence_delta=attribution.convergence_delta,
            parameters=attribution.parameters,
            warnings=attribution.warnings,
        )
