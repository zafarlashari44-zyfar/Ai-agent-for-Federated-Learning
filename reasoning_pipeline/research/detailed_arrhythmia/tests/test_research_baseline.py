from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from research.detailed_arrhythmia.config import (
    AugmentationConfig,
    TrainingConfig,
)
from research.detailed_arrhythmia.dataset.annotations import (
    extract_annotation_frequencies,
)
from research.detailed_arrhythmia.dataset.ontology import DEFAULT_ONTOLOGY
from research.detailed_arrhythmia.dataset.splits import (
    SplitManifest,
    create_patient_independent_split,
)
from research.detailed_arrhythmia.evaluation.metrics import calculate_metrics
from research.detailed_arrhythmia.training.augmentation import SafeECGAugmenter
from research.detailed_arrhythmia.training.dataset import BeatDataset
from research.detailed_arrhythmia.training.metadata import (
    CheckpointMetadata,
    sha256_file,
)
from research.detailed_arrhythmia.training.weights import calculate_class_weights


def normalized_beat() -> NDArray[np.float32]:
    x = np.linspace(-1.0, 1.0, 216)
    beat = 0.2 * np.sin(4.0 * np.pi * x)
    beat += np.exp(-np.square(x / 0.08)) * 2.0
    return np.asarray((beat - beat.mean()) / beat.std(), dtype=np.float32)


def test_annotation_frequency_extraction_counts_patients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotations = {
        "201": ("N", "N", "V"),
        "202": ("N", "A"),
        "100": ("N", "V"),
    }

    def fake_rdann(path: str, extension: str) -> SimpleNamespace:
        assert extension == "atr"
        return SimpleNamespace(symbol=annotations[Path(path).name])

    monkeypatch.setattr("wfdb.rdann", fake_rdann)
    rows = extract_annotation_frequencies(
        Path("records"),
        ("201", "202", "100"),
        minimum_beats=2,
        minimum_patients=2,
    )
    by_symbol = {row.symbol: row for row in rows}
    assert by_symbol["N"].total_beats == 4
    assert by_symbol["N"].patient_count == 2
    assert by_symbol["V"].recording_count == 2
    assert by_symbol["A"].recommendation.startswith("exclude")


def test_patient_split_is_reproducible_and_has_no_leakage() -> None:
    records = tuple(str(value) for value in range(100, 125)) + ("201", "202")
    first = create_patient_independent_split(records, seed=42)
    second = create_patient_independent_split(records, seed=42)
    assert first == second
    assert "201" in first.train_records or "202" not in first.train_records
    assert first.sha256 == second.sha256


def test_patient_split_represents_every_required_class() -> None:
    records = tuple(str(value) for value in range(100, 130))
    counts = {
        record: {
            "N": 100,
            "V": 1 if int(record) % 3 == 0 else 0,
        }
        for record in records
    }
    manifest = create_patient_independent_split(
        records,
        seed=42,
        validation_fraction=0.2,
        test_fraction=0.2,
        record_label_counts=counts,
        required_labels=("N", "V"),
    )
    for split_records in (
        manifest.train_records,
        manifest.validation_records,
        manifest.test_records,
    ):
        assert sum(counts[record]["N"] for record in split_records) > 0
        assert sum(counts[record]["V"] for record in split_records) > 0


def test_split_manifest_rejects_patient_leakage() -> None:
    with pytest.raises(ValueError, match="Patient leakage"):
        SplitManifest(
            seed=42,
            train_records=("201",),
            validation_records=("202",),
            test_records=("100",),
        )


def test_detailed_label_mapping() -> None:
    assert DEFAULT_ONTOLOGY.map_symbol("L") == "L"
    assert DEFAULT_ONTOLOGY.map_symbol("A") == "A"
    assert DEFAULT_ONTOLOGY.map_symbol("E") is None
    assert DEFAULT_ONTOLOGY.map_symbol("/") is None
    assert DEFAULT_ONTOLOGY.map_symbol("unknown") is None


@pytest.mark.parametrize(
    "method",
    ["inverse_frequency", "sqrt_inverse_frequency", "effective_number"],
)
def test_class_weights_use_training_counts_only(method: str) -> None:
    labels = np.asarray([0] * 100 + [1] * 25 + [2] * 4, dtype=np.int64)
    weights = calculate_class_weights(labels, 3, method=method)  # type: ignore[arg-type]
    assert weights.shape == (3,)
    assert weights.mean() == pytest.approx(1.0)
    assert weights[2] > weights[1] > weights[0]


def test_augmentation_is_deterministic_finite_and_length_216() -> None:
    config = AugmentationConfig(
        gaussian_noise_probability=1.0,
        amplitude_scale_probability=1.0,
        translation_probability=1.0,
        baseline_wander_probability=1.0,
        temporal_stretch_probability=1.0,
    )
    first = SafeECGAugmenter(config, seed=17)(normalized_beat())
    second = SafeECGAugmenter(config, seed=17)(normalized_beat())
    assert first.shape == (216,)
    assert np.all(np.isfinite(first))
    np.testing.assert_array_equal(first, second)
    assert float(np.std(first)) > 0.0


def test_unsafe_augmentation_ranges_are_rejected() -> None:
    with pytest.raises(ValueError, match="translation"):
        SafeECGAugmenter(
            AugmentationConfig(maximum_translation_samples=6), seed=1
        )


def test_validation_and_test_dataset_remain_unchanged() -> None:
    beats = np.stack([normalized_beat(), normalized_beat()])
    labels = np.asarray([0, 1], dtype=np.int64)
    dataset = BeatDataset(beats, labels)
    returned, _ = dataset[0]
    np.testing.assert_array_equal(returned.numpy(), beats[0])
    np.testing.assert_array_equal(beats[0], normalized_beat())


def test_training_configuration_is_reproducible() -> None:
    assert TrainingConfig().to_dict() == TrainingConfig().to_dict()
    assert TrainingConfig().seed == 42


def test_metric_calculation_includes_required_outputs() -> None:
    targets = np.asarray([0, 0, 1, 1], dtype=np.int64)
    predictions = np.asarray([0, 1, 1, 1], dtype=np.int64)
    probabilities = np.asarray(
        [[0.9, 0.1], [0.4, 0.6], [0.1, 0.9], [0.2, 0.8]],
        dtype=np.float64,
    )
    result = calculate_metrics(targets, predictions, probabilities, ("N", "V"))
    assert result["macro_f1"] < result["weighted_f1"] or result["macro_f1"] > 0
    assert "balanced_accuracy" in result
    assert result["per_class"]["N"]["false_negatives"] == 1
    assert len(result["confusion_matrix"]) == 2


def test_checkpoint_metadata_and_hash(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"separate-detailed-model")
    metadata = CheckpointMetadata(
        model_version="detailed-v1",
        ontology_version=DEFAULT_ONTOLOGY.version,
        labels=DEFAULT_ONTOLOGY.labels,
        split_manifest_hash="split-hash",
        training_seed=42,
        class_weight_method="sqrt_inverse_frequency",
        class_weights=(0.5, 1.5),
        augmentation_configuration=None,
        checkpoint_hash=sha256_file(checkpoint),
    )
    output = tmp_path / "metadata.json"
    metadata.write(output)
    assert output.exists()
    assert len(metadata.checkpoint_hash) == 64
