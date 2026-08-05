from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from research.detailed_arrhythmia.config import AugmentationConfig

FloatArray = NDArray[np.float32]


class SafeECGAugmenter:
    def __init__(self, config: AugmentationConfig, *, seed: int) -> None:
        self.config = config
        self.rng = np.random.default_rng(seed)
        self._validate_config()

    def __call__(self, beat: FloatArray) -> FloatArray:
        original = np.asarray(beat, dtype=np.float32)
        if original.shape != (216,):
            raise ValueError("ECG augmentation requires exactly 216 samples")
        augmented = original.astype(np.float64, copy=True)
        scale = max(float(np.std(original)), self.config.minimum_standard_deviation)
        if self.rng.random() < self.config.gaussian_noise_probability:
            relative = self.rng.uniform(*self.config.gaussian_noise_std_range)
            augmented += self.rng.normal(0.0, relative * scale, augmented.size)
        if self.rng.random() < self.config.amplitude_scale_probability:
            augmented *= self.rng.uniform(*self.config.amplitude_scale_range)
        if self.rng.random() < self.config.translation_probability:
            shift = int(
                self.rng.integers(
                    -self.config.maximum_translation_samples,
                    self.config.maximum_translation_samples + 1,
                )
            )
            augmented = self._translate(augmented, shift)
        if self.rng.random() < self.config.baseline_wander_probability:
            amplitude = self.rng.uniform(
                0.0, self.config.maximum_baseline_wander_ratio
            ) * scale
            phase = self.rng.uniform(0.0, 2.0 * np.pi)
            cycles = self.rng.uniform(0.5, 1.5)
            augmented += amplitude * np.sin(
                np.linspace(phase, phase + cycles * 2.0 * np.pi, 216)
            )
        if self.rng.random() < self.config.temporal_stretch_probability:
            augmented = self._stretch(
                augmented,
                self.rng.uniform(*self.config.temporal_stretch_range),
            )
        self._check_safety(original, augmented)
        return augmented.astype(np.float32)

    @staticmethod
    def _translate(values: NDArray[np.float64], shift: int) -> NDArray[np.float64]:
        if shift == 0:
            return values
        translated = np.empty_like(values)
        if shift > 0:
            translated[:shift] = values[0]
            translated[shift:] = values[:-shift]
        else:
            translated[shift:] = values[-1]
            translated[:shift] = values[-shift:]
        return translated

    @staticmethod
    def _stretch(values: NDArray[np.float64], factor: float) -> NDArray[np.float64]:
        stretched_length = max(2, round(values.size * factor))
        stretched = np.interp(
            np.linspace(0.0, values.size - 1, stretched_length),
            np.arange(values.size),
            values,
        )
        return np.interp(
            np.linspace(0.0, stretched_length - 1, values.size),
            np.arange(stretched_length),
            stretched,
        )

    def _check_safety(
        self,
        original: FloatArray,
        augmented: NDArray[np.float64],
    ) -> None:
        if augmented.shape != (216,) or not np.all(np.isfinite(augmented)):
            raise ValueError("Augmented beat must be finite and length 216")
        if float(np.std(augmented)) < self.config.minimum_standard_deviation:
            raise ValueError("Augmentation produced a flat or constant beat")
        if float(np.max(np.abs(augmented))) > self.config.maximum_absolute_amplitude:
            raise ValueError("Augmented beat exceeds the amplitude safety limit")
        original_energy = float(np.sum(np.square(original[60:156])))
        augmented_energy = float(np.sum(np.square(augmented[60:156])))
        ratio = augmented_energy / max(original_energy, 1e-12)
        lower, upper = self.config.central_qrs_energy_ratio_range
        if not lower <= ratio <= upper:
            raise ValueError("Augmentation changed central QRS energy excessively")

    def _validate_config(self) -> None:
        if not 0.005 <= self.config.gaussian_noise_std_range[0] <= 0.02:
            raise ValueError("Gaussian noise range is outside the safe contract")
        if not 0.005 <= self.config.gaussian_noise_std_range[1] <= 0.02:
            raise ValueError("Gaussian noise range is outside the safe contract")
        if (
            self.config.amplitude_scale_range[0] < 0.90
            or self.config.amplitude_scale_range[1] > 1.10
        ):
            raise ValueError("Amplitude scaling is outside the safe contract")
        if self.config.maximum_translation_samples > 5:
            raise ValueError("Temporal translation cannot exceed five samples")
        if self.config.maximum_baseline_wander_ratio > 0.05:
            raise ValueError("Baseline wander exceeds the safe contract")
        if (
            self.config.temporal_stretch_range[0] < 0.95
            or self.config.temporal_stretch_range[1] > 1.05
        ):
            raise ValueError("Temporal stretching is outside the safe contract")
