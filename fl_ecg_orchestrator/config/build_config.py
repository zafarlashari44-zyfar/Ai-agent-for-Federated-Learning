from __future__ import annotations

from collections import Counter
from math import ceil
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

PARTITION_OPTIMIZATION_ITERATIONS = 60000
PARTITION_OPTIMIZATION_RESTARTS = 12
PARTITION_OPTIMIZATION_TARGET_SCORE = 95.0
PARTITION_OPTIMIZATION_START_TEMPERATURE = 1.5
PARTITION_OPTIMIZATION_END_TEMPERATURE = 0.0005
PARTITION_LNS_ATTEMPTS = 12000

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
    if not record_ids:
        counts = np.zeros(
            NUM_CLASSES,
            dtype=np.int64,
        )
    else:
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


def get_class_record_map(
    records: list[str],
    metadata: dict[str, dict[str, Any]],
) -> dict[int, list[str]]:
    class_record_map: dict[int, list[str]] = {
        class_id: []
        for class_id in range(NUM_CLASSES)
    }

    for record_id in records:
        for class_id in metadata[
            record_id
        ]["present_classes"]:
            class_record_map[class_id].append(
                record_id
            )

    for class_id in range(NUM_CLASSES):
        class_record_map[class_id] = sorted(
            class_record_map[class_id],
            key=int,
        )

    return class_record_map


def get_client_counts(
    client_records: list[str],
    metadata: dict[str, dict[str, Any]],
) -> np.ndarray:
    if not client_records:
        return np.zeros(
            NUM_CLASSES,
            dtype=np.int64,
        )

    return np.sum(
        [
            metadata[record_id]["counts"]
            for record_id in client_records
        ],
        axis=0,
    ).astype(np.int64)


def calculate_record_priority(
    record_id: str,
    metadata: dict[str, dict[str, Any]],
    class_rarity: Counter,
) -> tuple[float, int, int]:
    rarity_score = sum(
        1.0 / max(
            class_rarity[class_id],
            1,
        )
        for class_id in metadata[
            record_id
        ]["present_classes"]
    )

    minority_beats = int(
        metadata[record_id]["counts"][1:].sum()
    )

    return (
        rarity_score,
        minority_beats,
        metadata[record_id]["num_beats"],
    )


def calculate_assignment_score(
    client_records: list[str],
    candidate_record: str,
    metadata: dict[str, dict[str, Any]],
    target_records_per_client: float,
    target_beats_per_client: float,
    target_class_counts: np.ndarray,
    coverage_required: set[int],
) -> float:
    current_counts = get_client_counts(
        client_records,
        metadata,
    )

    candidate_counts = metadata[
        candidate_record
    ]["counts"]

    projected_counts = (
        current_counts
        + candidate_counts
    )

    projected_records = (
        len(client_records)
        + 1
    )

    projected_beats = int(
        projected_counts.sum()
    )

    record_balance_penalty = (
        abs(
            projected_records
            - target_records_per_client
        )
        / max(
            target_records_per_client,
            1.0,
        )
    )

    beat_balance_penalty = (
        abs(
            projected_beats
            - target_beats_per_client
        )
        / max(
            target_beats_per_client,
            1.0,
        )
    )

    class_scale = np.maximum(
        target_class_counts,
        1.0,
    )

    class_balance_penalty = float(
        np.mean(
            np.abs(
                projected_counts
                - target_class_counts
            )
            / class_scale
        )
    )

    present_before = {
        class_id
        for class_id, count in enumerate(current_counts)
        if count > 0
    }

    present_after = {
        class_id
        for class_id, count in enumerate(projected_counts)
        if count > 0
    }

    missing_required_after = (
        coverage_required
        - present_after
    )

    new_required_classes = (
        present_after
        - present_before
    ) & coverage_required

    coverage_penalty = (
        len(missing_required_after)
        * 20.0
    )

    new_class_bonus = (
        len(new_required_classes)
        * 5.0
    )

    minority_target = np.maximum(
        target_class_counts[1:],
        1.0,
    )

    minority_balance_penalty = float(
        np.mean(
            np.abs(
                projected_counts[1:]
                - target_class_counts[1:]
            )
            / minority_target
        )
    )

    return (
        coverage_penalty
        + record_balance_penalty * 2.0
        + beat_balance_penalty * 3.0
        + class_balance_penalty * 4.0
        + minority_balance_penalty * 6.0
        - new_class_bonus
    )


def seed_required_class_coverage(
    clients: list[list[str]],
    assigned_records: set[str],
    training_records: list[str],
    metadata: dict[str, dict[str, Any]],
    class_record_map: dict[int, list[str]],
    coverage_required: set[int],
    target_records_per_client: float,
    target_beats_per_client: float,
    target_class_counts: np.ndarray,
) -> None:
    ordered_classes = sorted(
        coverage_required,
        key=lambda class_id: (
            len(class_record_map[class_id]),
            class_id,
        ),
    )

    for class_id in ordered_classes:
        for client_index in range(NUM_CLIENTS):
            client_counts = get_client_counts(
                clients[client_index],
                metadata,
            )

            if client_counts[class_id] > 0:
                continue

            candidates = [
                record_id
                for record_id in class_record_map[class_id]
                if record_id not in assigned_records
            ]

            if not candidates:
                raise RuntimeError(
                    "Unable to satisfy required class coverage for "
                    f"class {CLASS_NAMES[class_id]} on client "
                    f"{client_index + 1}"
                )

            best_record = min(
                candidates,
                key=lambda record_id: (
                    calculate_assignment_score(
                        client_records=clients[client_index],
                        candidate_record=record_id,
                        metadata=metadata,
                        target_records_per_client=(
                            target_records_per_client
                        ),
                        target_beats_per_client=(
                            target_beats_per_client
                        ),
                        target_class_counts=(
                            target_class_counts
                        ),
                        coverage_required=(
                            coverage_required
                        ),
                    ),
                    -len(
                        metadata[
                            record_id
                        ]["present_classes"]
                    ),
                    int(record_id),
                ),
            )

            clients[client_index].append(
                best_record
            )

            assigned_records.add(
                best_record
            )


def assign_remaining_records(
    clients: list[list[str]],
    assigned_records: set[str],
    training_records: list[str],
    metadata: dict[str, dict[str, Any]],
    coverage_required: set[int],
    target_records_per_client: float,
    target_beats_per_client: float,
    target_class_counts: np.ndarray,
) -> None:
    class_rarity = Counter()

    for record_id in training_records:
        for class_id in metadata[
            record_id
        ]["present_classes"]:
            class_rarity[class_id] += 1

    remaining_records = [
        record_id
        for record_id in training_records
        if record_id not in assigned_records
    ]

    ordered_records = sorted(
        remaining_records,
        key=lambda record_id: (
            calculate_record_priority(
                record_id,
                metadata,
                class_rarity,
            ),
            -int(record_id),
        ),
        reverse=True,
    )

    maximum_records_per_client = ceil(
        len(training_records)
        / NUM_CLIENTS
    )

    for record_id in ordered_records:
        available_clients = [
            client_index
            for client_index in range(NUM_CLIENTS)
            if len(clients[client_index])
            < maximum_records_per_client
        ]

        if not available_clients:
            available_clients = list(
                range(NUM_CLIENTS)
            )

        best_client_index = min(
            available_clients,
            key=lambda client_index: (
                calculate_assignment_score(
                    client_records=clients[client_index],
                    candidate_record=record_id,
                    metadata=metadata,
                    target_records_per_client=(
                        target_records_per_client
                    ),
                    target_beats_per_client=(
                        target_beats_per_client
                    ),
                    target_class_counts=(
                        target_class_counts
                    ),
                    coverage_required=(
                        coverage_required
                    ),
                ),
                len(clients[client_index]),
                client_index,
            ),
        )

        clients[best_client_index].append(
            record_id
        )

        assigned_records.add(
            record_id
        )



def partition_has_required_coverage(
    clients: list[list[str]],
    metadata: dict[str, dict[str, Any]],
    coverage_required: set[int],
) -> bool:
    for client_records in clients:
        counts = get_client_counts(
            client_records,
            metadata,
        )

        for class_id in coverage_required:
            if counts[class_id] <= 0:
                return False

    return True


def optimize_client_partition(
    clients: list[list[str]],
    training_records: list[str],
    metadata: dict[str, dict[str, Any]],
) -> tuple[list[list[str]], dict[str, Any]]:
    """
    Hybrid whole-patient optimizer.

    Search stages:
    1. Multiple simulated-annealing restarts
    2. Pair-swap local search
    3. Three-client cyclic swaps
    4. Large-neighbourhood random reconstruction

    Every move keeps complete patient records intact. The optimizer never
    duplicates, splits, removes, or leaks a patient between clients.
    """
    class_record_map = get_class_record_map(
        records=training_records,
        metadata=metadata,
    )

    coverage_required = {
        class_id
        for class_id in range(NUM_CLASSES)
        if len(class_record_map[class_id]) >= NUM_CLIENTS
    }

    def clone_partition(
        partition: list[list[str]],
    ) -> list[list[str]]:
        return [
            list(client_records)
            for client_records in partition
        ]

    def is_valid(
        partition: list[list[str]],
    ) -> bool:
        if any(
            len(client_records)
            != len(clients[0])
            for client_records in partition
        ):
            return False

        flattened = [
            record_id
            for client_records in partition
            for record_id in client_records
        ]

        if len(flattened) != len(set(flattened)):
            return False

        if set(flattened) != set(training_records):
            return False

        return partition_has_required_coverage(
            clients=partition,
            metadata=metadata,
            coverage_required=coverage_required,
        )

    def score_partition(
        partition: list[list[str]],
    ) -> tuple[float, dict[str, Any]]:
        quality = calculate_partition_quality(
            clients=partition,
            metadata=metadata,
        )

        # Use a higher-resolution internal objective than the displayed
        # two-decimal overall score. Minority balance is the main remaining
        # bottleneck, while beat balance acts as a tie-breaker.
        score = (
            float(quality["overall_score"])
            + float(quality["minority_balance_score"]) * 1e-4
            + float(quality["beat_balance_score"]) * 1e-6
        )

        return score, quality

    base_partition = clone_partition(clients)
    base_score, base_quality = score_partition(
        base_partition
    )

    global_best = clone_partition(
        base_partition
    )
    global_best_score = base_score
    global_best_quality = dict(base_quality)

    total_accepted = 0
    total_improving = 0
    total_coverage_rejections = 0
    total_iterations = 0
    restart_scores: list[float] = []

    for restart in range(
        PARTITION_OPTIMIZATION_RESTARTS
    ):
        rng = np.random.default_rng(
            SEED + 2026 + restart * 1009
        )

        if restart == 0:
            current = clone_partition(
                base_partition
            )
        else:
            # Produce a diverse valid restart by applying random valid swaps
            # to the strongest solution found so far.
            current = clone_partition(
                global_best
            )

            warmup_swaps = 250 + restart * 25

            for _ in range(warmup_swaps):
                client_a, client_b = rng.choice(
                    NUM_CLIENTS,
                    size=2,
                    replace=False,
                )

                index_a = int(
                    rng.integers(
                        len(current[client_a])
                    )
                )

                index_b = int(
                    rng.integers(
                        len(current[client_b])
                    )
                )

                current[client_a][index_a], current[client_b][index_b] = (
                    current[client_b][index_b],
                    current[client_a][index_a],
                )

                if not partition_has_required_coverage(
                    clients=[
                        current[client_a],
                        current[client_b],
                    ],
                    metadata=metadata,
                    coverage_required=coverage_required,
                ):
                    current[client_a][index_a], current[client_b][index_b] = (
                        current[client_b][index_b],
                        current[client_a][index_a],
                    )

        current_score, current_quality = score_partition(
            current
        )

        restart_best = clone_partition(
            current
        )
        restart_best_score = current_score
        restart_best_quality = dict(
            current_quality
        )

        for iteration in range(
            PARTITION_OPTIMIZATION_ITERATIONS
        ):
            total_iterations += 1

            if (
                float(
                    global_best_quality[
                        "overall_score"
                    ]
                )
                >= PARTITION_OPTIMIZATION_TARGET_SCORE
            ):
                break

            progress = (
                iteration
                / max(
                    PARTITION_OPTIMIZATION_ITERATIONS - 1,
                    1,
                )
            )

            temperature = (
                PARTITION_OPTIMIZATION_START_TEMPERATURE
                * (
                    PARTITION_OPTIMIZATION_END_TEMPERATURE
                    / PARTITION_OPTIMIZATION_START_TEMPERATURE
                )
                ** progress
            )

            # Most moves are pair swaps. Some are 3-client cyclic swaps,
            # which can escape pair-swap local optima.
            use_cycle = bool(
                rng.random() < 0.18
            )

            if use_cycle:
                selected_clients = list(
                    rng.choice(
                        NUM_CLIENTS,
                        size=3,
                        replace=False,
                    )
                )

                selected_indices = [
                    int(
                        rng.integers(
                            len(current[client_id])
                        )
                    )
                    for client_id in selected_clients
                ]

                original_records = [
                    current[client_id][record_index]
                    for client_id, record_index in zip(
                        selected_clients,
                        selected_indices,
                    )
                ]

                current[
                    selected_clients[0]
                ][selected_indices[0]] = original_records[2]

                current[
                    selected_clients[1]
                ][selected_indices[1]] = original_records[0]

                current[
                    selected_clients[2]
                ][selected_indices[2]] = original_records[1]

                affected = [
                    current[client_id]
                    for client_id in selected_clients
                ]

                move_description = (
                    selected_clients,
                    selected_indices,
                    original_records,
                )
            else:
                client_a, client_b = rng.choice(
                    NUM_CLIENTS,
                    size=2,
                    replace=False,
                )

                index_a = int(
                    rng.integers(
                        len(current[client_a])
                    )
                )

                index_b = int(
                    rng.integers(
                        len(current[client_b])
                    )
                )

                current[client_a][index_a], current[client_b][index_b] = (
                    current[client_b][index_b],
                    current[client_a][index_a],
                )

                affected = [
                    current[client_a],
                    current[client_b],
                ]

                move_description = (
                    int(client_a),
                    int(client_b),
                    index_a,
                    index_b,
                )

            coverage_valid = (
                partition_has_required_coverage(
                    clients=affected,
                    metadata=metadata,
                    coverage_required=coverage_required,
                )
            )

            if not coverage_valid:
                total_coverage_rejections += 1

                if use_cycle:
                    (
                        selected_clients,
                        selected_indices,
                        original_records,
                    ) = move_description

                    for (
                        client_id,
                        record_index,
                        original_record,
                    ) in zip(
                        selected_clients,
                        selected_indices,
                        original_records,
                    ):
                        current[
                            client_id
                        ][record_index] = original_record
                else:
                    (
                        client_a,
                        client_b,
                        index_a,
                        index_b,
                    ) = move_description

                    current[client_a][index_a], current[client_b][index_b] = (
                        current[client_b][index_b],
                        current[client_a][index_a],
                    )

                continue

            candidate_score, candidate_quality = score_partition(
                current
            )

            delta = candidate_score - current_score

            accept = delta >= 0.0

            if not accept:
                probability = float(
                    np.exp(
                        delta
                        / max(
                            temperature,
                            1e-12,
                        )
                    )
                )

                accept = bool(
                    rng.random() < probability
                )

            if accept:
                total_accepted += 1

                if delta > 0.0:
                    total_improving += 1

                current_score = candidate_score
                current_quality = candidate_quality

                if current_score > restart_best_score:
                    restart_best = clone_partition(
                        current
                    )
                    restart_best_score = current_score
                    restart_best_quality = dict(
                        current_quality
                    )

                if current_score > global_best_score:
                    global_best = clone_partition(
                        current
                    )
                    global_best_score = current_score
                    global_best_quality = dict(
                        current_quality
                    )
            else:
                if use_cycle:
                    (
                        selected_clients,
                        selected_indices,
                        original_records,
                    ) = move_description

                    for (
                        client_id,
                        record_index,
                        original_record,
                    ) in zip(
                        selected_clients,
                        selected_indices,
                        original_records,
                    ):
                        current[
                            client_id
                        ][record_index] = original_record
                else:
                    (
                        client_a,
                        client_b,
                        index_a,
                        index_b,
                    ) = move_description

                    current[client_a][index_a], current[client_b][index_b] = (
                        current[client_b][index_b],
                        current[client_a][index_a],
                    )

        restart_scores.append(
            round(
                float(
                    restart_best_quality[
                        "overall_score"
                    ]
                ),
                2,
            )
        )

        if (
            float(
                global_best_quality[
                    "overall_score"
                ]
            )
            >= PARTITION_OPTIMIZATION_TARGET_SCORE
        ):
            break

    # Large-neighbourhood search: randomly select one record from four
    # clients and test multiple permutations of those whole records.
    lns_rng = np.random.default_rng(
        SEED + 99001
    )

    lns_improvements = 0

    from itertools import permutations

    for _ in range(
        PARTITION_LNS_ATTEMPTS
    ):
        if (
            float(
                global_best_quality[
                    "overall_score"
                ]
            )
            >= PARTITION_OPTIMIZATION_TARGET_SCORE
        ):
            break

        selected_clients = list(
            lns_rng.choice(
                NUM_CLIENTS,
                size=4,
                replace=False,
            )
        )

        selected_indices = [
            int(
                lns_rng.integers(
                    len(global_best[client_id])
                )
            )
            for client_id in selected_clients
        ]

        original_records = [
            global_best[client_id][record_index]
            for client_id, record_index in zip(
                selected_clients,
                selected_indices,
            )
        ]

        best_local_partition = None
        best_local_score = global_best_score
        best_local_quality = None

        for permuted_records in permutations(
            original_records
        ):
            if list(permuted_records) == original_records:
                continue

            candidate = clone_partition(
                global_best
            )

            for (
                client_id,
                record_index,
                record_id,
            ) in zip(
                selected_clients,
                selected_indices,
                permuted_records,
            ):
                candidate[
                    client_id
                ][record_index] = record_id

            if not partition_has_required_coverage(
                clients=[
                    candidate[client_id]
                    for client_id in selected_clients
                ],
                metadata=metadata,
                coverage_required=coverage_required,
            ):
                continue

            candidate_score, candidate_quality = score_partition(
                candidate
            )

            if candidate_score > best_local_score:
                best_local_partition = candidate
                best_local_score = candidate_score
                best_local_quality = candidate_quality

        if best_local_partition is not None:
            global_best = best_local_partition
            global_best_score = best_local_score
            global_best_quality = dict(
                best_local_quality
            )
            lns_improvements += 1

    global_best = [
        sorted(
            client_records,
            key=int,
        )
        for client_records in global_best
    ]

    if not is_valid(global_best):
        raise ValueError(
            "Hybrid optimization produced an invalid partition"
        )

    optimization_report = {
        "method": (
            "multi_start_simulated_annealing_"
            "cyclic_swaps_large_neighbourhood_search"
        ),
        "seed": SEED + 2026,
        "target_score": (
            PARTITION_OPTIMIZATION_TARGET_SCORE
        ),
        "restarts_configured": (
            PARTITION_OPTIMIZATION_RESTARTS
        ),
        "restarts_completed": len(
            restart_scores
        ),
        "iterations_per_restart": (
            PARTITION_OPTIMIZATION_ITERATIONS
        ),
        "total_iterations": total_iterations,
        "accepted_moves": total_accepted,
        "improving_moves": total_improving,
        "rejected_coverage_moves": (
            total_coverage_rejections
        ),
        "lns_attempts": (
            PARTITION_LNS_ATTEMPTS
        ),
        "lns_improvements": (
            lns_improvements
        ),
        "restart_best_scores": (
            restart_scores
        ),
        "initial_score": round(
            float(
                base_quality[
                    "overall_score"
                ]
            ),
            2,
        ),
        "best_score": round(
            float(
                global_best_quality[
                    "overall_score"
                ]
            ),
            2,
        ),
        "score_improvement": round(
            float(
                global_best_quality[
                    "overall_score"
                ]
            )
            - float(
                base_quality[
                    "overall_score"
                ]
            ),
            2,
        ),
        "target_reached": bool(
            float(
                global_best_quality[
                    "overall_score"
                ]
            )
            >= PARTITION_OPTIMIZATION_TARGET_SCORE
        ),
        "best_quality": (
            global_best_quality
        ),
        "constraints_preserved": {
            "whole_patient_assignment": True,
            "equal_patient_count": True,
            "no_duplicate_patients": True,
            "no_patient_leakage": True,
            "required_class_coverage": True,
        },
    }

    return (
        global_best,
        optimization_report,
    )



def assign_training_records_to_clients(
    training_records: list[str],
    metadata: dict[str, dict[str, Any]],
) -> list[list[str]]:
    clients = [
        []
        for _ in range(NUM_CLIENTS)
    ]

    assigned_records: set[str] = set()

    class_record_map = get_class_record_map(
        records=training_records,
        metadata=metadata,
    )

    coverage_required = {
        class_id
        for class_id in range(NUM_CLASSES)
        if len(class_record_map[class_id])
        >= NUM_CLIENTS
    }

    total_class_counts = np.sum(
        [
            metadata[record_id]["counts"]
            for record_id in training_records
        ],
        axis=0,
    ).astype(np.float64)

    total_training_beats = float(
        total_class_counts.sum()
    )

    target_records_per_client = (
        len(training_records)
        / NUM_CLIENTS
    )

    target_beats_per_client = (
        total_training_beats
        / NUM_CLIENTS
    )

    target_class_counts = (
        total_class_counts
        / NUM_CLIENTS
    )

    seed_required_class_coverage(
        clients=clients,
        assigned_records=assigned_records,
        training_records=training_records,
        metadata=metadata,
        class_record_map=class_record_map,
        coverage_required=coverage_required,
        target_records_per_client=(
            target_records_per_client
        ),
        target_beats_per_client=(
            target_beats_per_client
        ),
        target_class_counts=target_class_counts,
    )

    assign_remaining_records(
        clients=clients,
        assigned_records=assigned_records,
        training_records=training_records,
        metadata=metadata,
        coverage_required=coverage_required,
        target_records_per_client=(
            target_records_per_client
        ),
        target_beats_per_client=(
            target_beats_per_client
        ),
        target_class_counts=target_class_counts,
    )

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

    validate_client_class_coverage(
        clients=clients,
        metadata=metadata,
        class_record_map=class_record_map,
    )

    return clients


def validate_client_class_coverage(
    clients: list[list[str]],
    metadata: dict[str, dict[str, Any]],
    class_record_map: dict[int, list[str]],
) -> None:
    for class_id in range(NUM_CLASSES):
        records_with_class = len(
            class_record_map[class_id]
        )

        clients_with_class = 0

        for client_records in clients:
            counts = get_client_counts(
                client_records,
                metadata,
            )

            if counts[class_id] > 0:
                clients_with_class += 1

        if (
            records_with_class >= NUM_CLIENTS
            and clients_with_class != NUM_CLIENTS
        ):
            raise ValueError(
                f"Class {CLASS_NAMES[class_id]} could appear on all "
                f"clients but appears on only {clients_with_class}"
            )

        if records_with_class < NUM_CLIENTS:
            print(
                "Coverage warning: "
                f"class {CLASS_NAMES[class_id]} appears in only "
                f"{records_with_class} training patient records, "
                "so full five-client coverage is impossible."
            )


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



def calculate_partition_quality(
    clients: list[list[str]],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    patient_counts = np.asarray(
        [
            len(client_records)
            for client_records in clients
        ],
        dtype=np.float64,
    )

    client_class_counts = np.asarray(
        [
            get_client_counts(
                client_records,
                metadata,
            )
            for client_records in clients
        ],
        dtype=np.float64,
    )

    beat_counts = client_class_counts.sum(
        axis=1
    )

    target_patient_count = float(
        patient_counts.mean()
    )

    target_beat_count = float(
        beat_counts.mean()
    )

    patient_balance_score = 100.0 * (
        1.0
        - float(
            np.mean(
                np.abs(
                    patient_counts
                    - target_patient_count
                )
            )
        )
        / max(
            target_patient_count,
            1.0,
        )
    )

    beat_balance_score = 100.0 * (
        1.0
        - float(
            np.mean(
                np.abs(
                    beat_counts
                    - target_beat_count
                )
            )
        )
        / max(
            target_beat_count,
            1.0,
        )
    )

    coverage_scores = []

    for class_id in range(NUM_CLASSES):
        clients_with_class = int(
            np.sum(
                client_class_counts[
                    :,
                    class_id,
                ]
                > 0
            )
        )

        coverage_scores.append(
            clients_with_class
            / NUM_CLIENTS
        )

    class_coverage_score = (
        float(
            np.mean(
                coverage_scores
            )
        )
        * 100.0
    )

    minority_class_scores: dict[str, float] = {}

    for class_id in range(
        1,
        NUM_CLASSES,
    ):
        class_counts = client_class_counts[
            :,
            class_id,
        ]

        total_class_beats = float(
            class_counts.sum()
        )

        if total_class_beats <= 0:
            minority_score = 100.0
        else:
            uniform_target = np.full(
                NUM_CLIENTS,
                total_class_beats
                / NUM_CLIENTS,
                dtype=np.float64,
            )

            total_variation_distance = (
                float(
                    np.abs(
                        class_counts
                        - uniform_target
                    ).sum()
                )
                / (
                    2.0
                    * total_class_beats
                )
            )

            minority_score = (
                1.0
                - total_variation_distance
            ) * 100.0

        minority_class_scores[
            CLASS_NAMES[class_id]
        ] = round(
            max(
                0.0,
                min(
                    100.0,
                    minority_score,
                ),
            ),
            2,
        )

    minority_balance_score = float(
        np.mean(
            list(
                minority_class_scores.values()
            )
        )
    )

    flattened_records = [
        record_id
        for client_records in clients
        for record_id in client_records
    ]

    no_duplicate_records = (
        len(flattened_records)
        == len(
            set(flattened_records)
        )
    )

    leakage_score = (
        100.0
        if no_duplicate_records
        else 0.0
    )

    weights = {
        "patient_balance": 0.15,
        "beat_balance": 0.20,
        "class_coverage": 0.25,
        "minority_balance": 0.30,
        "integrity": 0.10,
    }

    overall_score = (
        patient_balance_score
        * weights["patient_balance"]
        + beat_balance_score
        * weights["beat_balance"]
        + class_coverage_score
        * weights["class_coverage"]
        + minority_balance_score
        * weights["minority_balance"]
        + leakage_score
        * weights["integrity"]
    )

    if overall_score >= 90.0:
        quality_grade = "Excellent"
    elif overall_score >= 80.0:
        quality_grade = "Good"
    elif overall_score >= 70.0:
        quality_grade = "Acceptable"
    elif overall_score >= 60.0:
        quality_grade = "Needs improvement"
    else:
        quality_grade = "Poor"

    return {
        "overall_score": round(
            overall_score,
            2,
        ),
        "quality_grade": quality_grade,
        "patient_balance_score": round(
            max(
                0.0,
                min(
                    100.0,
                    patient_balance_score,
                ),
            ),
            2,
        ),
        "beat_balance_score": round(
            max(
                0.0,
                min(
                    100.0,
                    beat_balance_score,
                ),
            ),
            2,
        ),
        "class_coverage_score": round(
            max(
                0.0,
                min(
                    100.0,
                    class_coverage_score,
                ),
            ),
            2,
        ),
        "minority_balance_score": round(
            max(
                0.0,
                min(
                    100.0,
                    minority_balance_score,
                ),
            ),
            2,
        ),
        "minority_class_scores": (
            minority_class_scores
        ),
        "record_integrity_score": (
            leakage_score
        ),
        "duplicate_patient_records": (
            not no_duplicate_records
        ),
        "weights": weights,
    }


def build_partition_audit(
    clients: list[list[str]],
    metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    client_audit: dict[str, Any] = {}

    for index, client_records in enumerate(
        clients,
        start=1,
    ):
        counts = get_client_counts(
            client_records,
            metadata,
        )

        total_beats = int(
            counts.sum()
        )

        percentages = {
            CLASS_NAMES[class_id]: round(
                (
                    float(
                        counts[class_id]
                    )
                    / max(
                        total_beats,
                        1,
                    )
                )
                * 100.0,
                4,
            )
            for class_id in range(
                NUM_CLASSES
            )
        }

        client_audit[
            f"client_{index}"
        ] = {
            "num_patients": len(
                client_records
            ),
            "num_beats": total_beats,
            "class_counts": {
                CLASS_NAMES[class_id]: int(
                    counts[class_id]
                )
                for class_id in range(
                    NUM_CLASSES
                )
            },
            "class_percentages": (
                percentages
            ),
            "present_classes": [
                CLASS_NAMES[class_id]
                for class_id in range(
                    NUM_CLASSES
                )
                if counts[class_id] > 0
            ],
        }

    return {
        "clients": client_audit,
        "quality": (
            calculate_partition_quality(
                clients=clients,
                metadata=metadata,
            )
        ),
    }


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

    clients, optimization_report = optimize_client_partition(
        clients=clients,
        training_records=training_records,
        metadata=metadata,
    )

    validate_partition(
        clients=clients,
        global_test_records=global_test_records,
        all_records=records,
    )

    partition_audit = build_partition_audit(
        clients=clients,
        metadata=metadata,
    )

    partition_audit[
        "optimization"
    ] = optimization_report

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
        "partition_audit": partition_audit,
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
    print("=" * 70)
    print("GLOBAL TEST PARTITION")
    print("=" * 70)

    print(
        f"Records: {global_test_records}"
    )

    print(
        "Distribution: "
        f"{calculate_class_distribution(global_test_records, metadata)}"
    )

    print()
    print("=" * 70)
    print("CLIENT PARTITION AUDIT")
    print("=" * 70)

    for client_id, client_records in (
        config["clients"].items()
    ):
        audit = config["partition_audit"]["clients"][client_id]

        print()
        print(client_id.upper())
        print("-" * 70)

        print(
            f"Patient records ({audit['num_patients']}): "
            f"{client_records}"
        )

        print(
            f"Total beats: {audit['num_beats']}"
        )

        print(
            f"Class counts: {audit['class_counts']}"
        )

        print(
            f"Class percentages: "
            f"{audit['class_percentages']}"
        )

        print(
            f"Present classes: "
            f"{audit['present_classes']}"
        )

    optimization = config[
        "partition_audit"
    ]["optimization"]

    print()
    print("=" * 70)
    print("PARTITION OPTIMIZATION")
    print("=" * 70)

    print(
        "Method: "
        f"{optimization['method']}"
    )

    print(
        "Initial score: "
        f"{optimization['initial_score']}%"
    )

    print(
        "Optimized score: "
        f"{optimization['best_score']}%"
    )

    print(
        "Improvement: "
        f"{optimization['score_improvement']} points"
    )

    print(
        "Iterations completed: "
        f"{optimization['total_iterations']}"
    )

    print(
        "Accepted swaps: "
        f"{optimization['accepted_moves']}"
    )

    print(
        "Improving swaps: "
        f"{optimization['improving_moves']}"
    )

    print(
        "Target reached: "
        f"{optimization['target_reached']}"
    )

    print(
        "Restarts completed: "
        f"{optimization['restarts_completed']}"
    )

    print(
        "Restart best scores: "
        f"{optimization['restart_best_scores']}"
    )

    print(
        "Large-neighbourhood improvements: "
        f"{optimization['lns_improvements']}"
    )

    quality = config["partition_audit"]["quality"]

    print()
    print("=" * 70)
    print("PARTITION QUALITY SCORE")
    print("=" * 70)

    print(
        "Overall score: "
        f"{quality['overall_score']}%"
    )

    print(
        "Quality grade: "
        f"{quality['quality_grade']}"
    )

    print(
        "Patient balance: "
        f"{quality['patient_balance_score']}%"
    )

    print(
        "Beat balance: "
        f"{quality['beat_balance_score']}%"
    )

    print(
        "Class coverage: "
        f"{quality['class_coverage_score']}%"
    )

    print(
        "Minority balance: "
        f"{quality['minority_balance_score']}%"
    )

    print(
        "Minority-class scores: "
        f"{quality['minority_class_scores']}"
    )

    print(
        "Record integrity: "
        f"{quality['record_integrity_score']}%"
    )

    print()
    print("=" * 70)
    print("PARTITION VALIDATION PASSED")
    print("=" * 70)

    print(
        "No patient leakage detected."
    )

    print(
        "All records were assigned exactly once."
    )

    print(
        "All feasible classes were distributed across every client."
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