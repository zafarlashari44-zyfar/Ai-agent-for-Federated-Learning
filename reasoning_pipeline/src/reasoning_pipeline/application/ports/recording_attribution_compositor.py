from typing import Protocol

from reasoning_pipeline.domain.models.recording_attribution_overlay import (
    RecordingAttributionOverlay,
)
from reasoning_pipeline.domain.models.recording_explanation import (
    RecordingExplanation,
)


class RecordingAttributionCompositorProtocol(Protocol):
    def compose(
        self,
        *,
        total_source_samples: int,
        sampling_rate_hz: float,
        recording_explanation: RecordingExplanation,
        method_id: str,
    ) -> RecordingAttributionOverlay:
        ...
