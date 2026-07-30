from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from reasoning_pipeline.application.ports.beat_explainer import (
    LocalAttribution,
)
from reasoning_pipeline.domain.models.attribution_map import AttributionMap

if TYPE_CHECKING:
    from reasoning_pipeline.orchestration.model_input_preparer import PreparedBeat


class AttributionMapperProtocol(Protocol):
    def map_to_source(
        self,
        *,
        prepared_beat: PreparedBeat,
        attribution: LocalAttribution,
        target_label: str,
    ) -> AttributionMap:
        ...
