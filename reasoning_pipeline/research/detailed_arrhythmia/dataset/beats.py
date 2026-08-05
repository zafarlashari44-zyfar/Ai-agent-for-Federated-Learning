from pathlib import Path

import numpy as np
import wfdb
from numpy.typing import NDArray

from reasoning_pipeline.orchestration.model_input_preparer import ModelInputPreparer
from research.detailed_arrhythmia.dataset.ontology import DetailedLabelOntology

FloatArray = NDArray[np.float32]


def extract_expert_annotated_beats(
    records_dir: Path,
    record_ids: tuple[str, ...],
    ontology: DetailedLabelOntology,
    *,
    lead_name: str = "MLII",
) -> tuple[FloatArray, NDArray[np.int64], tuple[str, ...]]:
    beats: list[NDArray[np.float32]] = []
    labels: list[int] = []
    sources: list[str] = []
    label_indices = {label: index for index, label in enumerate(ontology.labels)}
    for record_id in record_ids:
        record = wfdb.rdrecord(str(records_dir / record_id))
        annotation = wfdb.rdann(str(records_dir / record_id), "atr")
        names = tuple(record.sig_name)
        selected = (
            lead_name
            if lead_name in names
            else "II" if "II" in names else names[0]
        )
        signal = np.asarray(
            record.p_signal[:, names.index(selected)], dtype=np.float64
        )
        for peak, symbol in zip(annotation.sample, annotation.symbol, strict=True):
            mapped = ontology.map_symbol(symbol)
            if mapped is None:
                continue
            start = int(peak) - ModelInputPreparer.PRE_R_SAMPLES
            stop = int(peak) + ModelInputPreparer.POST_R_SAMPLES
            if start < 0 or stop > signal.size:
                continue
            beat = signal[start:stop]
            standard_deviation = float(np.std(beat))
            if not np.isfinite(standard_deviation) or standard_deviation <= 1e-8:
                continue
            normalized = (
                (beat - np.mean(beat)) / standard_deviation
            ).astype(np.float32)
            if normalized.shape != (ModelInputPreparer.INPUT_LENGTH,):
                continue
            beats.append(normalized)
            labels.append(label_indices[mapped])
            sources.append(record_id)
    return (
        np.asarray(beats, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        tuple(sources),
    )
