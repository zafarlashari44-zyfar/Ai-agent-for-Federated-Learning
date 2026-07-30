from dataclasses import dataclass

from reasoning_pipeline.domain.models.recording_attribution_overlay import (
    RecordingAttributionOverlay,
)
from reasoning_pipeline.domain.models.recording_attribution_point import (
    RecordingAttributionPoint,
)
from reasoning_pipeline.domain.models.recording_explanation import (
    RecordingExplanation,
)


@dataclass(frozen=True)
class _Contribution:
    attribution: float
    beat_index: int


class RecordingAttributionCompositor:
    """Compose beat attribution maps into a dense source-record overlay."""

    DEFAULT_AGGREGATION_METHOD = "maximum"

    def compose(
        self,
        *,
        total_source_samples: int,
        sampling_rate_hz: float,
        recording_explanation: RecordingExplanation,
        method_id: str,
    ) -> RecordingAttributionOverlay:
        if total_source_samples <= 0:
            raise ValueError("total_source_samples must be greater than zero")
        if sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be greater than zero")
        if not method_id.strip():
            raise ValueError("method_id cannot be empty")

        contributions: list[list[_Contribution]] = [
            []
            for _ in range(total_source_samples)
        ]
        method_versions: set[str] = set()
        explained_beats: set[int] = set()

        for beat_explanation in recording_explanation.beat_explanations:
            matching_maps = tuple(
                attribution_map
                for attribution_map in beat_explanation.attribution_maps
                if attribution_map.method_id == method_id
            )
            if len(matching_maps) > 1:
                raise ValueError(
                    f"Beat {beat_explanation.beat_index} has duplicate "
                    f"{method_id} attribution maps"
                )
            if not matching_maps:
                continue

            attribution_map = matching_maps[0]
            if attribution_map.sampling_rate_hz != sampling_rate_hz:
                raise ValueError(
                    "Attribution map sampling rate does not match recording"
                )

            method_versions.add(attribution_map.method_version)
            explained_beats.add(beat_explanation.beat_index)

            for point in attribution_map.points:
                source_index = point.source_sample_index
                if not 0 <= source_index < total_source_samples:
                    raise ValueError(
                        "Mapped attribution point exceeds recording bounds"
                    )
                if not 0.0 <= point.attribution <= 1.0:
                    raise ValueError(
                        "Grad-CAM attribution must be between 0 and 1"
                    )
                contributions[source_index].append(
                    _Contribution(
                        attribution=point.attribution,
                        beat_index=beat_explanation.beat_index,
                    )
                )

        if not explained_beats:
            raise ValueError(
                f"Recording explanation has no {method_id} attribution maps"
            )
        if len(method_versions) != 1:
            raise ValueError(
                "Selected attribution method must use one method version"
            )

        points = tuple(
            self._aggregate_point(
                source_index=source_index,
                sampling_rate_hz=sampling_rate_hz,
                contributions=sample_contributions,
            )
            for source_index, sample_contributions in enumerate(contributions)
        )

        return RecordingAttributionOverlay(
            record_id=recording_explanation.record_id,
            method_id=method_id,
            method_version=next(iter(method_versions)),
            sampling_rate_hz=sampling_rate_hz,
            total_source_samples=total_source_samples,
            aggregation_method=self.DEFAULT_AGGREGATION_METHOD,
            points=points,
            explained_beat_count=len(explained_beats),
            warnings=recording_explanation.warnings,
        )

    def _aggregate_point(
        self,
        *,
        source_index: int,
        sampling_rate_hz: float,
        contributions: list[_Contribution],
    ) -> RecordingAttributionPoint:
        if not contributions:
            return RecordingAttributionPoint(
                source_sample_index=source_index,
                timestamp_seconds=source_index / sampling_rate_hz,
                maximum_attribution=0.0,
                mean_attribution=0.0,
                coverage_count=0,
                contributing_beat_indices=(),
            )

        values = tuple(
            contribution.attribution
            for contribution in contributions
        )
        beat_indices = tuple(
            sorted(
                {
                    contribution.beat_index
                    for contribution in contributions
                }
            )
        )
        if len(beat_indices) != len(contributions):
            raise ValueError(
                "A beat may contribute only once to each source sample"
            )

        return RecordingAttributionPoint(
            source_sample_index=source_index,
            timestamp_seconds=source_index / sampling_rate_hz,
            maximum_attribution=max(values),
            mean_attribution=sum(values) / len(values),
            coverage_count=len(values),
            contributing_beat_indices=beat_indices,
        )
