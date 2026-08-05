from reasoning_pipeline.baseline_adapter.cnn1d import ECGCNN1D


def create_detailed_model(num_classes: int) -> ECGCNN1D:
    """Reuse the architecture, never the frozen production checkpoint."""
    return ECGCNN1D(input_length=216, num_classes=num_classes)
