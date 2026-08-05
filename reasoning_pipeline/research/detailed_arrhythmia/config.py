from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class AugmentationConfig:
    gaussian_noise_probability: float = 0.30
    gaussian_noise_std_range: tuple[float, float] = (0.005, 0.02)
    amplitude_scale_probability: float = 0.30
    amplitude_scale_range: tuple[float, float] = (0.90, 1.10)
    translation_probability: float = 0.20
    maximum_translation_samples: int = 5
    baseline_wander_probability: float = 0.20
    maximum_baseline_wander_ratio: float = 0.05
    temporal_stretch_probability: float = 0.20
    temporal_stretch_range: tuple[float, float] = (0.95, 1.05)
    maximum_absolute_amplitude: float = 8.0
    minimum_standard_deviation: float = 1e-6
    central_qrs_energy_ratio_range: tuple[float, float] = (0.40, 1.75)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 8
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    early_stopping_patience: int = 3
    class_weight_method: str = "sqrt_inverse_frequency"
    effective_number_beta: float = 0.9999
    input_length: int = 216
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
