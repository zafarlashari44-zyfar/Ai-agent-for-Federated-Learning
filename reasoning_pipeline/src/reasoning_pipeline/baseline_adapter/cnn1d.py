import torch
from torch import nn


class ECGCNN1D(nn.Module):

    def __init__(
        self,
        input_length: int = 216,
        num_classes: int = 5,
    ):
        super().__init__()

        self.input_length = input_length
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.ndim == 2:
            x = x.unsqueeze(1)

        if x.ndim != 3:
            raise ValueError(
                f"Expected tensor shape batch by 216 "
                f"or batch by 1 by 216, received {tuple(x.shape)}"
            )

        features = self.features(x)
        logits: torch.Tensor = self.classifier(features)

        return logits


def create_model(
    input_length: int = 216,
    num_classes: int = 5,
) -> ECGCNN1D:

    return ECGCNN1D(
        input_length=input_length,
        num_classes=num_classes,
    )
