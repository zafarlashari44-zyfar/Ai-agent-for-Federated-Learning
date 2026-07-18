from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RECORDS_DIR = PROJECT_ROOT / "datasets" / "records"

OUTPUT_PATH = (
    PROJECT_ROOT
    / "fl_ecg_orchestrator"
    / "config"
    / "config.yaml"
)

SEED = 42
NUM_CLIENTS = 5
NUM_GLOBAL_TEST_RECORDS = 8
NUM_CLASSES = 5
NUMBER_OF_TEST_TRIALS = 20000

LABEL_MAP = {
    "N": 0,
    "S": 1,
    "V": 2,
    "F": 3,
    "Q": 4,
}

CLASS_NAMES = {
    0: "N",
    1: "S",
    2: "V",
    3: "F",
    4: "Q",
}


def encode_labels(
    raw_labels: np.ndarray,
) -> np.ndarray:
    encoded = []

    for raw_label in raw_labels.reshape(-1):
        label = raw_label

        if isinstance(label, bytes):
            label = label.decode("utf-8")

        label = str(label).strip()

        if label not in LABEL_MAP:
            raise ValueError(
                f"Unsupported ECG label {label}"
            )

        encoded.append(
            LABEL_MAP[label]
        )

    return np.asarray(
        encoded,
        dtype=np.int64,
    )


def discover_records() -> list[str]:
    if not RECORDS_DIR.exists():
        raise FileNotFoundError(
            f"Records directory not found at {RECORDS_DIR}"
        )

    records = []

    for beats_path in RECORDS_DIR.glob(
        "data_mitdb_*_beats.npy"
    ):
        record_id = (
            beats_path.stem
            .replace("data_mitdb_", "")
            .replace("_beats", "")
        )

        labels_path = (
            RECORDS_DIR
            / f"data_mitdb_{record_id}_labels.npy"
        )

        if labels_path.exists():
            records.append(record_id)

    records = sorted(
        records,
        key=int,
    )

    if len(records) != 48:
        raise ValueError(
            f"Expected 48 complete records but found {len(records)}"
        )

    return records


def load_record_class_counts(
    record_id: str,
) -> np.ndarray:
    labels_path = (
        RECORDS_DIR
        / f"data_mitdb_{record_id}_labels.npy"
    )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Missing labels file for record {record_id}"
        )

    raw_labels = np.load(
        labels_path,
        allow_pickle=True,
    )

    labels = encode_labels(
        raw_labels
    )

    return np.bincount(
        labels,
        minlength=NUM_CLASSES,
    ).astype(np.int64)


def build_record_metadata(
    records: list[str],
) -> dict[str, dict[str, Any]]:
    metadata = {}

    for record_id in records:
        counts = load_record_class_counts(
            record_id
        )

        metadata[record_id] = {
            "counts": counts,
            "num_beats": int(counts.sum()),
            "present_classes": {
                class_id
                for class_id, count in enumerate(counts)
                if count > 0
            },
        }

    return metadata


def calculate_class_distribution(
    record_ids: list[str],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, int]:
    counts = np.sum(
        [
            metadata[record_id]["counts"]
            for record_id in record_ids
        ],
        axis=0,
    )

    return {
        CLASS_NAMES[class_id]: int(
            counts[class_id]
        )
        for class_id in range(NUM_CLASSES)
    }


def score_global_test_candidate(
    candidate: tuple[str, ...],
    metadata: dict[str, dict[str, Any]],
    global_counts: np.ndarray,
) -> float:
    candidate_counts = np.sum(
        [
            metadata[record_id]["counts"]
            for record_id in candidate
        ],
        axis=0,
    )

    missing_classes = int(
        np.sum(
            np.logical_and(
                global_counts > 0,
                candidate_counts == 0,
            )
        )
    )

    global_distribution = (
        global_counts
        / max(global_counts.sum(), 1)
    )

    candidate_distribution = (
        candidate_counts
        / max(candidate_counts.sum(), 1)
    )

    distribution_distance = float(
        np.abs(
            candidate_distribution
            - global_distribution
        ).sum()
    )

    minority_bonus = 0.0

    for class_id in range(1, NUM_CLASSES):
        if candidate_counts[class_id] > 0:
            minority_bonus += 0.1

    return (
        missing_classes * 100.0
        + distribution_distance
        - minority_bonus
    )


def choose_global_test_records(
    records: list[str],
    metadata: dict[str, dict[str, Any]],
) -> list[str]:
    global_counts = np.sum(
        [
            metadata[record_id]["counts"]
            for record_id in records
        ],
        axis=0,
    )

    rng = np.random.default_rng(SEED)

    records_array = np.asarray(records)

    best_score = float("inf")
    best_candidate: tuple[str, ...] | None = None

    for _ in range(NUMBER_OF_TEST_TRIALS):
        sampled = rng.choice(
            records_array,
            size=NUM_GLOBAL_TEST_RECORDS,
            replace=False,
        )

        candidate = tuple(
            str(record_id)
            for record_id in sampled
        )

        candidate_counts = np.sum(
            [
                metadata[record_id]["counts"]
                for record_id in candidate
            ],
            axis=0,
        )

        if np.any(
            np.logical_and(
                global_counts > 0,
                candidate_counts == 0,
            )
        ):
            continue

        score = score_global_test_candidate(
            candidate=candidate,
            metadata=metadata,
            global_counts=global_counts,
        )

        if score < best_score:
            best_score = score
            best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError(
            "Unable to find a global test set containing all classes"
        )

    selected = sorted(
        best_candidate,
        key=int,
    )

    test_distribution = calculate_class_distribution(
        selected,
        metadata,
    )

    missing_classes = [
        class_name
        for class_name, count in test_distribution.items()
        if count == 0
    ]

    if missing_classes:
        raise ValueError(
            f"Global test set is missing classes {missing_classes}"
        )

    return selected


def calculate_record_priority(
    record_id: str,
    metadata: dict[str, dict[str, Any]],
    class_rarity: Counter,
) -> tuple[float, int]:
    rarity_score = sum(
        1.0 / max(
            class_rarity[class_id],
            1,
        )
        for class_id in metadata[
            record_id
        ]["present_classes"]
    )

    return (
        rarity_score,
        metadata[record_id]["num_beats"],
    )


def client_assignment_score(
    client_records: list[str],
    candidate_record: str,
    metadata: dict[str, dict[str, Any]],
    target_records_per_client: int,
    target_beats_per_client: float,
) -> float:
    candidate_records = (
        client_records
        + [candidate_record]
    )

    counts = np.sum(
        [
            metadata[record_id]["counts"]
            for record_id in candidate_records
        ],
        axis=0,
    )

    present_classes = int(
        np.sum(counts > 0)
    )

    missing_class_penalty = float(
        NUM_CLASSES - present_classes
    ) * 8.0

    record_count_penalty = abs(
        len(candidate_records)
        - target_records_per_client
    )

    total_beats = int(
        counts.sum()
    )

    beat_balance_penalty = abs(
        total_beats
        - target_beats_per_client
    ) / max(
        target_beats_per_client,
        1.0,
    )

    existing_classes = set()

    for record_id in client_records:
        existing_classes.update(
            metadata[
                record_id
            ]["present_classes"]
        )

    new_classes = (
        metadata[
            candidate_record
        ]["present_classes"]
        - existing_classes
    )

    new_class_bonus = float(
        len(new_classes)
    ) * 3.0

    return (
        missing_class_penalty
        + record_count_penalty
        + beat_balance_penalty
        - new_class_bonus
    )


def assign_training_records_to_clients(
    training_records: list[str],
    metadata: dict[str, dict[str, Any]],
) -> list[list[str]]:
    clients = [
        []
        for _ in range(NUM_CLIENTS)
    ]

    class_rarity = Counter()

    for record_id in training_records:
        for class_id in metadata[
            record_id
        ]["present_classes"]:
            class_rarity[class_id] += 1

    ordered_records = sorted(
        training_records,
        key=lambda record_id: calculate_record_priority(
            record_id,
            metadata,
            class_rarity,
        ),
        reverse=True,
    )

    target_records_per_client = (
        len(training_records)
        // NUM_CLIENTS
    )

    total_training_beats = sum(
        metadata[record_id]["num_beats"]
        for record_id in training_records
    )

    target_beats_per_client = (
        total_training_beats
        / NUM_CLIENTS
    )

    for record_id in ordered_records:
        available_clients = [
            index
            for index, client_records in enumerate(clients)
            if len(client_records)
            < target_records_per_client
        ]

        if not available_clients:
            available_clients = list(
                range(NUM_CLIENTS)
            )

        best_client_index = min(
            available_clients,
            key=lambda index: client_assignment_score(
                client_records=clients[index],
                candidate_record=record_id,
                metadata=metadata,
                target_records_per_client=(
                    target_records_per_client
                ),
                target_beats_per_client=(
                    target_beats_per_client
                ),
            ),
        )

        clients[
            best_client_index
        ].append(record_id)

    clients = [
        sorted(
            client_records,
            key=int,
        )
        for client_records in clients
    ]

    flattened_records = [
        record_id
        for client_records in clients
        for record_id in client_records
    ]

    if len(flattened_records) != len(
        set(flattened_records)
    ):
        raise ValueError(
            "A patient record appears in more than one client"
        )

    if set(flattened_records) != set(
        training_records
    ):
        raise ValueError(
            "Some training records were not assigned"
        )

    return clients


def validate_partition(
    clients: list[list[str]],
    global_test_records: list[str],
    all_records: list[str],
) -> None:
    client_records = [
        record_id
        for records in clients
        for record_id in records
    ]

    if set(client_records) & set(
        global_test_records
    ):
        raise ValueError(
            "Global test records leaked into client training data"
        )

    assigned_records = set(
        client_records
    ) | set(
        global_test_records
    )

    if assigned_records != set(
        all_records
    ):
        raise ValueError(
            "Some records were not assigned"
        )

    if len(client_records) != len(
        set(client_records)
    ):
        raise ValueError(
            "A patient appears in multiple clients"
        )


def build_config() -> dict[str, Any]:
    records = discover_records()

    metadata = build_record_metadata(
        records
    )

    global_test_records = choose_global_test_records(
        records=records,
        metadata=metadata,
    )

    global_test_set = set(
        global_test_records
    )

    training_records = [
        record_id
        for record_id in records
        if record_id not in global_test_set
    ]

    clients = assign_training_records_to_clients(
        training_records=training_records,
        metadata=metadata,
    )

    validate_partition(
        clients=clients,
        global_test_records=global_test_records,
        all_records=records,
    )

    return {
        "project": {
            "name": (
                "Agent 2 Federated ECG Orchestrator"
            ),
            "seed": SEED,
            "records_dir": (
                "datasets/records"
            ),
            "output_dir": (
                "fl_ecg_orchestrator/outputs"
            ),
        },
        "data": {
            "input_length": 216,
            "num_classes": NUM_CLASSES,
            "label_map": LABEL_MAP,
            "global_test_records": (
                global_test_records
            ),
        },
        "clients": {
            f"client_{index + 1}": (
                client_records
            )
            for index, client_records in enumerate(clients)
        },
        "training": {
            "federated_rounds": 10,
            "local_epochs": 2,
            "batch_size": 128,
            "learning_rate": 0.001,
            "validation_ratio": 0.2,
        },
        "strategy": {
            "name": "fedavg",
            "proximal_mu": 0.0,
            "smotetomek": False,
        },
        "ablation": {
            "seeds": [
                42,
                52,
                62,
            ],
            "experiments": [
                {
                    "name": "fedavg",
                    "strategy": "fedavg",
                    "proximal_mu": 0.0,
                    "smotetomek": False,
                },
                {
                    "name": (
                        "fedavg_smotetomek"
                    ),
                    "strategy": "fedavg",
                    "proximal_mu": 0.0,
                    "smotetomek": True,
                },
                {
                    "name": "fedprox",
                    "strategy": "fedprox",
                    "proximal_mu_values": [
                        0.001,
                        0.01,
                        0.05,
                        0.1,
                        0.5,
                    ],
                    "smotetomek": False,
                },
                {
                    "name": (
                        "fedprox_smotetomek"
                    ),
                    "strategy": "fedprox",
                    "proximal_mu_values": [
                        0.001,
                        0.01,
                        0.05,
                        0.1,
                        0.5,
                    ],
                    "smotetomek": True,
                },
            ],
        },
    }


def save_config(
    config: dict[str, Any],
) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
        )


def print_summary(
    config: dict[str, Any],
) -> None:
    records = discover_records()

    metadata = build_record_metadata(
        records
    )

    global_test_records = (
        config["data"][
            "global_test_records"
        ]
    )

    print()
    print("Global test records")
    print(global_test_records)

    print("Global test distribution")
    print(
        calculate_class_distribution(
            record_ids=global_test_records,
            metadata=metadata,
        )
    )

    for client_id, client_records in (
        config["clients"].items()
    ):
        print()
        print(client_id)
        print(client_records)
        print(
            calculate_class_distribution(
                record_ids=client_records,
                metadata=metadata,
            )
        )


def main() -> None:
    config = build_config()

    save_config(
        config
    )

    print(
        f"Configuration saved to {OUTPUT_PATH}"
    )

    print_summary(
        config
    )


if __name__ == "__main__":
    main()