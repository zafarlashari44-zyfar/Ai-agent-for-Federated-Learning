from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn


class LIME1D:
    """
    LIME style temporal region explainer for 1D ECG beats.

    The CNN always receives 216 samples.

    The interpretable representation consists of 54 binary
    temporal regions. A value of 1 keeps the original region.
    A value of 0 replaces that region with the masking baseline.
    """

    METHOD_ID = "lime-temporal-mask-1d"
    METHOD_VERSION = "2.0.0"

    INPUT_LENGTH = 216
    NUM_REGIONS = 54
    SAMPLES_PER_REGION = 4
    NUM_CLASSES = 5

    def __init__(
        self,
        *,
        model: nn.Module,
        num_samples: int = 1000,
        random_state: int = 42,
        masking_value: float = 0.0,
        kernel_width: float = 0.25,
        mask_probability: float = 0.10,
    ) -> None:

        if num_samples < 2:
            raise ValueError(
                "num_samples must be at least 2"
            )

        if kernel_width <= 0:
            raise ValueError(
                "kernel_width must be positive"
            )

        if not 0.0 < mask_probability < 1.0:
            raise ValueError(
                "mask_probability must be between 0 and 1"
            )

        self.model = model
        self.num_samples = num_samples
        self.random_state = random_state
        self.masking_value = float(masking_value)
        self.kernel_width = float(kernel_width)
        self.mask_probability = float(mask_probability)

        try:
            self.device = next(
                model.parameters()
            ).device
        except StopIteration:
            self.device = torch.device("cpu")

        self.model.eval()

    def _validate_samples(
        self,
        samples: Sequence[float],
    ) -> np.ndarray:

        array = np.asarray(
            samples,
            dtype=np.float32,
        )

        if array.shape != (
            self.INPUT_LENGTH,
        ):
            raise ValueError(
                "Expected 216 ECG samples"
            )

        if not np.all(
            np.isfinite(array)
        ):
            raise ValueError(
                "LIME input contains non finite values"
            )

        return array

    def _generate_masks(
        self,
    ) -> np.ndarray:
        """
        Generate local binary perturbations.

        A value of 1 preserves the original ECG region.
        A value of 0 masks the region.

        Row 0 always represents the untouched original beat.

        Every remaining perturbation is guaranteed to mask
        at least one temporal region.
        """

        rng = np.random.default_rng(
            self.random_state
        )

        masked = rng.random(
            (
                self.num_samples,
                self.NUM_REGIONS,
            )
        ) < self.mask_probability

        # Row 0 is deliberately the original beat.
        masked[0, :] = False

        # Every other perturbation must differ from the
        # original beat by at least one region.
        empty_rows = np.where(
            ~masked[1:].any(axis=1)
        )[0] + 1

        if len(empty_rows) > 0:
            forced_regions = rng.integers(
                low=0,
                high=self.NUM_REGIONS,
                size=len(empty_rows),
            )

            masked[
                empty_rows,
                forced_regions,
            ] = True

        masks = (
            ~masked
        ).astype(np.float32)

        return masks

    def _apply_masks(
        self,
        samples: np.ndarray,
        masks: np.ndarray,
    ) -> np.ndarray:
        """
        Convert region masks into 216 sample ECG inputs.

        Regions marked 1 retain the original waveform.
        Regions marked 0 are replaced by masking_value.
        """

        if masks.ndim != 2:
            raise ValueError(
                "Masks must be two dimensional"
            )

        if masks.shape[1] != self.NUM_REGIONS:
            raise ValueError(
                "Expected 54 mask regions"
            )

        expanded_masks = np.repeat(
            masks,
            self.SAMPLES_PER_REGION,
            axis=1,
        )

        if expanded_masks.shape[1] != self.INPUT_LENGTH:
            raise RuntimeError(
                "Expanded mask length is not 216"
            )

        original = samples.reshape(
            1,
            self.INPUT_LENGTH,
        )

        perturbed = (
            expanded_masks * original
            + (1.0 - expanded_masks)
            * self.masking_value
        )

        return perturbed.astype(
            np.float32
        )

    def _predict_proba(
        self,
        samples: np.ndarray,
    ) -> np.ndarray:

        tensor = torch.tensor(
            samples,
            dtype=torch.float32,
            device=self.device,
        )

        self.model.eval()

        with torch.no_grad():
            logits = self.model(
                tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

        return (
            probabilities
            .detach()
            .cpu()
            .numpy()
        )

    def _calculate_distances(
        self,
        masks: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate normalized Hamming distance from
        the original all present representation.
        """

        distances = np.mean(
            1.0 - masks,
            axis=1,
        )

        return distances.astype(
            np.float64
        )

    def _kernel(
        self,
        distances: np.ndarray,
    ) -> np.ndarray:
        """
        Give greater weight to perturbations that remain
        close to the original ECG beat.
        """

        weights = np.exp(
            -(
                distances ** 2
            )
            / (
                self.kernel_width ** 2
            )
        )

        return weights.astype(
            np.float64
        )

    def explain(
        self,
        *,
        samples: Sequence[float],
        target_class: int,
    ) -> np.ndarray:

        if not (
            0
            <= target_class
            < self.NUM_CLASSES
        ):
            raise ValueError(
                "target_class must be between 0 and 4"
            )

        original = self._validate_samples(
            samples
        )

        masks = self._generate_masks()

        perturbed_samples = self._apply_masks(
            original,
            masks,
        )

        probabilities = self._predict_proba(
            perturbed_samples
        )

        if probabilities.shape != (
            self.num_samples,
            self.NUM_CLASSES,
        ):
            raise RuntimeError(
                "Unexpected CNN probability shape"
            )

        target_probabilities = probabilities[
            :,
            target_class,
        ]

        distances = self._calculate_distances(
            masks
        )

        sample_weights = self._kernel(
            distances
        )

        surrogate = Ridge(
            alpha=1.0,
            fit_intercept=True,
        )

        surrogate.fit(
            masks,
            target_probabilities,
            sample_weight=sample_weights,
        )

        attribution = np.asarray(
            surrogate.coef_,
            dtype=np.float32,
        )

        if attribution.shape != (
            self.NUM_REGIONS,
        ):
            raise RuntimeError(
                "LIME surrogate did not produce "
                "54 region coefficients"
            )

        if not np.all(
            np.isfinite(attribution)
        ):
            raise RuntimeError(
                "LIME produced non finite attribution"
            )

        return attribution

    def verify_original_representation(
        self,
        samples: Sequence[float],
    ) -> np.ndarray:
        """
        Return the ECG represented by an all ones mask.

        This should be numerically identical to the original
        216 sample input.
        """

        original = self._validate_samples(
            samples
        )

        mask = np.ones(
            (
                1,
                self.NUM_REGIONS,
            ),
            dtype=np.float32,
        )

        represented = self._apply_masks(
            original,
            mask,
        )[0]

        return represented
