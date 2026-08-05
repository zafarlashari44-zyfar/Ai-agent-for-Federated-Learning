from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import wfdb

from research.detailed_arrhythmia.dataset.ontology import DetailedLabelOntology

BEAT_SYMBOLS = frozenset(
    {"N", "L", "R", "A", "a", "J", "S", "V", "F", "e", "j", "E", "/", "f", "Q"}
)


@dataclass(frozen=True)
class AnnotationFrequency:
    symbol: str
    total_beats: int
    patient_count: int
    recording_count: int
    percentage_of_all_annotations: float
    recommendation: str


def patient_id_for_record(record_id: str) -> str:
    """Return the known MIT-BIH subject identity for a recording."""
    return "201_202" if record_id in {"201", "202"} else record_id


def extract_annotation_frequencies(
    records_dir: Path,
    record_ids: tuple[str, ...],
    *,
    annotation_extension: str = "atr",
    minimum_beats: int = 500,
    minimum_patients: int = 4,
) -> tuple[AnnotationFrequency, ...]:
    counts: dict[str, int] = defaultdict(int)
    patients: dict[str, set[str]] = defaultdict(set)
    recordings: dict[str, set[str]] = defaultdict(set)
    for record_id in record_ids:
        annotation = wfdb.rdann(
            str(records_dir / record_id),
            annotation_extension,
        )
        patient_id = patient_id_for_record(record_id)
        for symbol in annotation.symbol:
            counts[symbol] += 1
            patients[symbol].add(patient_id)
            recordings[symbol].add(record_id)
    total = sum(counts.values())
    rows = []
    for symbol in sorted(counts):
        enough_beats = counts[symbol] >= minimum_beats
        enough_patients = len(patients[symbol]) >= minimum_patients
        if symbol not in BEAT_SYMBOLS:
            recommendation = "exclude: annotation is not a beat label"
        elif symbol == "/":
            recommendation = "exclude v1: paced beats occur in only four patients"
        elif enough_beats and enough_patients:
            recommendation = "include: sufficient beats and patient diversity"
        elif not enough_beats and not enough_patients:
            recommendation = "exclude: too few beats and too little patient diversity"
        elif not enough_beats:
            recommendation = "exclude: too few beats for reliable evaluation"
        else:
            recommendation = "exclude: insufficient patient diversity"
        rows.append(
            AnnotationFrequency(
                symbol=symbol,
                total_beats=counts[symbol],
                patient_count=len(patients[symbol]),
                recording_count=len(recordings[symbol]),
                percentage_of_all_annotations=(counts[symbol] / total * 100.0),
                recommendation=recommendation,
            )
        )
    return tuple(rows)


def write_frequency_reports(
    rows: tuple[AnnotationFrequency, ...],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row.__dict__ for row in rows])
    csv_path = output_dir / "annotation_frequencies.csv"
    markdown_path = output_dir / "annotation_frequencies.md"
    frame.to_csv(csv_path, index=False)
    headers = tuple(frame.columns)
    lines = [
        "# MIT-BIH annotation frequency report",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| "
        + " | ".join(str(value).replace("|", "\\|") for value in row)
        + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path


def extract_record_label_counts(
    records_dir: Path,
    record_ids: tuple[str, ...],
    ontology: DetailedLabelOntology,
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for record_id in record_ids:
        counts = {label: 0 for label in ontology.labels}
        annotation = wfdb.rdann(str(records_dir / record_id), "atr")
        for symbol in annotation.symbol:
            mapped = ontology.map_symbol(symbol)
            if mapped is not None:
                counts[mapped] += 1
        result[record_id] = counts
    return result
