from reasoning_pipeline.application.ports.beat_explainer import LocalAttribution


class NoOpBeatExplainer:
    """Placeholder explainer that deliberately produces no attribution."""

    @property
    def method_id(self) -> str:
        return "noop"

    def explain(
        self,
        *,
        samples: tuple[float, ...],
        target_class: int,
    ) -> LocalAttribution | None:
        return None
