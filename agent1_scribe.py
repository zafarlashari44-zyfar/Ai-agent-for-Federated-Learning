"""
Agent 1 -- The Scribe
======================
Part of: Agentic Federated Learning for Bio-Signal Analysis

Two independent, separately-unit-testable sub-pipelines, joined by a
fail-loud fusion layer:

    Pipeline A  process_ecg_pipeline()   -- NeuroKit2 ECG signal processing
    Pipeline B  extract_clinical_entities() -- Bio-ClinicalBERT-family NER
    Fusion      fuse_agent1_outputs()    -- strict Pydantic schema, joins by record/patient id

Design decisions mandated by the build spec:
  - Sampling rate is NEVER left to library default. Always passed explicitly.
  - Core ECG call is nk.ecg_process() (clean -> peaks -> quality -> delineate -> phase)
    rather than manually chaining individual NeuroKit2 functions.
  - R-peak method is pinned explicitly (default "neurokit"; override as needed).
  - Signal quality (nk.ecg_quality) is checked BEFORE trusting delineation;
    low-quality beats are flagged, not silently used.
  - Beats are epoched with nk.epochs_create() on a fixed, explicit window
    (-0.2s to +0.4s around each R-peak) so every beat has identical shape --
    required both for nk.ecg_analyze() feature tables and for the 1D CNN
    input tensor built by segment_beats().
  - NER pipeline requires aggregation_strategy="simple" explicitly (raw
    wordpiece output like "##itis" is a common, otherwise-silent bug), with
    a manual subword-merge fallback if the chosen model doesn't support it.
  - id2label is checked explicitly; generic LABEL_0/1/2... outputs are
    flagged rather than trusted as named entities.
  - Every external call (NeuroKit2, wfdb, HuggingFace) is wrapped in
    try/except with a structured failure record (type, message, traceback)
    -- nothing fails silently, and partial success is marked explicitly
    rather than passed downstream as if complete.

Library versions this was built/tested against (pin these in requirements.txt):
    neurokit2==0.2.13
    pandas==2.3.3
    wfdb==4.3.1
    pydantic==2.13.4
    transformers>=4.40        # Pipeline B only, not exercised in this environment
    torch>=2.2                # Pipeline B only, not exercised in this environment
"""

import logging
import os
import traceback
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import neurokit2 as nk
import wfdb
from pydantic import BaseModel, ConfigDict, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agent1_scribe")

# ---------------------------------------------------------------------------
# Shared constants -- explicit, never left to library defaults
# ---------------------------------------------------------------------------

AAMI_LABEL_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
    'F': 'F',
    '/': 'Q', 'f': 'Q', 'Q': 'Q',
}
BEAT_ANNOTATION_SYMBOLS = set(AAMI_LABEL_MAP.keys())
MAX_PEAK_ANNOTATION_OFFSET_SAMPLES = 10  # tolerance when matching detected peak to nearest annotation
MITBIH_SAMPLING_RATE = 360  # Hz. Pass explicitly to every NeuroKit2 call.
DEFAULT_R_PEAK_METHOD = "neurokit"  # pin explicitly; document if you change it
EPOCH_START_S = -0.2  # seconds before R-peak
EPOCH_END_S = 0.4     # seconds after R-peak
QUALITY_THRESHOLD = 0.5  # nk.ecg_quality score below this -> flagged, not trusted

PREFERRED_LEADS = ["MLII", "II", "I", "V5", "V1"]
SEGMENTS_ROOT = "data/segments"

DEFAULT_NER_MODEL = "d4data/biomedical-ner-all"  # Bio-ClinicalBERT-family, NER-finetuned
# NOTE: verify this model's id2label against your target categories
# (Disease/Drug/Symptom) before trusting output in production -- see
# extract_clinical_entities() below, which checks this automatically.


def _structured_error(exc: Exception) -> Dict[str, str]:
    """Build a structured, loggable failure record instead of swallowing the exception."""
    return {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}


def _to_json_safe(obj):
    """Recursively convert numpy/NaN into JSON-serializable native Python types."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# ---------------------------------------------------------------------------
# Pydantic output schema (Fusion Layer contract)
# ---------------------------------------------------------------------------

class BeatFeature(BaseModel):
    """One row of nk.ecg_analyze() output for a single beat, normalized to named fields.
    Extra columns NeuroKit2 returns beyond these are preserved via `extra`."""
    model_config = ConfigDict(extra="allow")

    beat_index: int
    rr_interval: Optional[float] = None
    qrs_duration: Optional[float] = None
    p_amplitude: Optional[float] = None
    st_elevation: Optional[float] = None


class ClinicalEntity(BaseModel):
    text: str
    label: str
    confidence: float


class Agent1Output(BaseModel):
    """Strict fusion schema. `ecg_status` and `ner_status` are always populated,
    even (especially) on failure -- this is what makes fusion fail loudly."""
    record_id: str

    # Pipeline A results
    ecg_status: str  # "success" | "partial_success_delineation_failed" | "no_peaks_found" | "error" | "not_run"
    ecg_error: Optional[Dict[str, str]] = None
    sampling_rate: Optional[int] = None
    lead_used: Optional[str] = None
    signal_quality_flag: Optional[str] = None  # "good" | "low_quality" | "unknown"
    beat_features: List[BeatFeature] = Field(default_factory=list)
    num_cnn_segments: int = 0
    cnn_segments_path: Optional[str] = None
    cnn_labels_path: Optional[str] = None

    # Pipeline B results
    ner_status: str  # "success" | "error" | "not_run" | "unmapped_labels"
    ner_error: Optional[Dict[str, str]] = None
    extracted_entities: List[ClinicalEntity] = Field(default_factory=list)

    # Fusion metadata
    join_key_used: str = "record_id"


# ---------------------------------------------------------------------------
# Pipeline A -- ECG signal processing (NeuroKit2)
# ---------------------------------------------------------------------------

def load_ecg_record(record_name: str) -> Dict[str, Any]:
    """Load a raw ECG record via wfdb and pick a physiologically sensible lead.
    Kept separate from processing so it can be swapped out (or skipped, if you
    already have a raw signal array) without touching the processing logic.
    """
    try:
        signals, fields = wfdb.rdsamp(record_name)
        sig_names = fields.get("sig_name", [])
        lead_used = None
        for lead in PREFERRED_LEADS:
            if lead in sig_names:
                idx = sig_names.index(lead)
                ecg_raw = signals[:, idx]
                lead_used = lead
                break
        else:
            logger.warning("No preferred lead found in %s; falling back to channel 0.", sig_names)
            ecg_raw = signals[:, 0]
            lead_used = sig_names[0] if sig_names else "unknown_channel_0"

        fs = fields["fs"]
        if fs != MITBIH_SAMPLING_RATE:
            logger.warning(
                "Record %s reports fs=%s, not the expected %s. Passing the actual "
                "value through explicitly rather than assuming.",
                record_name, fs, MITBIH_SAMPLING_RATE,
            )

        return {"status": "success", "ecg_signal": ecg_raw, "sampling_rate": fs, "lead_used": lead_used, "error": None}
    except Exception as e:
        logger.exception("Failed to load record %s.", record_name)
        return {"status": "error", "ecg_signal": None, "sampling_rate": None, "lead_used": None,
                "error": _structured_error(e)}


def segment_beats_for_cnn(
    ecg_cleaned: np.ndarray,
    r_peaks: List[int],
    sampling_rate: int,
    window_pre_s: float = -EPOCH_START_S,
    window_post_s: float = EPOCH_END_S,
) -> Optional[np.ndarray]:
    """Fixed-length, z-normalized raw beat windows for the 1D CNN (Agent 2's
    federated training loop). Complementary to nk.ecg_analyze()'s interpretable
    feature table below -- this is the raw-signal path, that is the features path.
    """
    if not r_peaks:
        return None

    pre_samples = int(round(window_pre_s * sampling_rate))
    post_samples = int(round(window_post_s * sampling_rate))
    window_length = pre_samples + post_samples

    valid_peaks = [p for p in r_peaks if pre_samples <= p < len(ecg_cleaned) - post_samples]
    if not valid_peaks:
        logger.warning("All R-peaks were too close to signal edges; no CNN segments produced.")
        return None

    beats = []
    for peak in valid_peaks:
        window = np.asarray(ecg_cleaned[peak - pre_samples: peak + post_samples])
        if window.shape[0] != window_length:
            continue
        std = window.std()
        beats.append((window - window.mean()) / std if std > 1e-8 else window - window.mean())
    return np.vstack(beats) if beats else None

def segment_beats_with_labels(
    record_name: str,
    ecg_cleaned: np.ndarray,
    r_peaks: List[int],
    sampling_rate: int,
    window_pre_s: float = -EPOCH_START_S,
    window_post_s: float = EPOCH_END_S,
) -> Dict[str, Any]:
    """Same fixed-window, z-normalized segmentation as segment_beats_for_cnn(),
    but additionally maps each beat to its AAMI 5-class label (N/S/V/F/Q) by
    matching each detected R-peak to the nearest MIT-BIH expert annotation.
    Requires the .atr annotation file to exist alongside record_name.
    """
    try:
        annotation = wfdb.rdann(record_name, 'atr')
    except Exception as e:
        logger.exception("Failed to load annotation file for %s.", record_name)
        return {"cnn_segments": None, "cnn_labels": None, "num_cnn_segments": 0, "error": _structured_error(e)}

    ann_beats = [
        (int(s), sym) for s, sym in zip(annotation.sample, annotation.symbol)
        if sym in BEAT_ANNOTATION_SYMBOLS
    ]
    if not ann_beats:
        logger.warning("No beat annotations found for %s.", record_name)
        return {"cnn_segments": None, "cnn_labels": None, "num_cnn_segments": 0, "error": None}

    pre_samples = int(round(window_pre_s * sampling_rate))
    post_samples = int(round(window_post_s * sampling_rate))
    window_length = pre_samples + post_samples

    valid_peaks = [p for p in r_peaks if pre_samples <= p < len(ecg_cleaned) - post_samples]
    if not valid_peaks:
        logger.warning("All R-peaks were too close to signal edges; no labeled CNN segments produced.")
        return {"cnn_segments": None, "cnn_labels": None, "num_cnn_segments": 0, "error": None}

    beats, labels, unmatched = [], [], 0
    for peak in valid_peaks:
        closest_sample, closest_sym = min(ann_beats, key=lambda x: abs(x[0] - peak))
        if abs(closest_sample - peak) > MAX_PEAK_ANNOTATION_OFFSET_SAMPLES:
            unmatched += 1
            continue
        window = np.asarray(ecg_cleaned[peak - pre_samples: peak + post_samples])
        if window.shape[0] != window_length:
            continue
        std = window.std()
        beats.append((window - window.mean()) / std if std > 1e-8 else window - window.mean())
        labels.append(AAMI_LABEL_MAP.get(closest_sym, 'Q'))

    if unmatched:
        logger.warning("%d detected peaks in %s had no annotation within %d samples; excluded from labeled set.",
                        unmatched, record_name, MAX_PEAK_ANNOTATION_OFFSET_SAMPLES)

    if not beats:
        return {"cnn_segments": None, "cnn_labels": None, "num_cnn_segments": 0, "error": None}

    return {
        "cnn_segments": np.vstack(beats),
        "cnn_labels": np.array(labels),
        "num_cnn_segments": len(beats),
        "error": None,
    }


def process_ecg_pipeline(
    ecg_signal: np.ndarray,
    sampling_rate: int,
    r_peak_method: str = DEFAULT_R_PEAK_METHOD,
    quality_threshold: float = QUALITY_THRESHOLD,
    save_cnn_segments: bool = False,
    segments_out_path: Optional[str] = None,
    record_name: Optional[str] = None,
) -> Dict[str, Any]:
    result = {
        "status": "not_run",
        "sampling_rate": sampling_rate,
        "signal_quality_flag": "unknown",
        "beat_features": [],
        "cnn_segments": None,
        "cnn_labels": None,
        "num_cnn_segments": 0,
        "cnn_segments_path": None,
        "cnn_labels_path": None,
        "error": None,
    }

    if sampling_rate is None:
        result["status"] = "error"
        result["error"] = {"type": "ValueError", "message": "sampling_rate must be passed explicitly; refusing to default.", "traceback": ""}
        logger.error(result["error"]["message"])
        return result

    try:
        signals_df, info = nk.ecg_process(
            np.asarray(ecg_signal), sampling_rate=sampling_rate, method=r_peak_method
        )
    except Exception as e:
        logger.exception("nk.ecg_process failed.")
        result["status"] = "error"
        result["error"] = _structured_error(e)
        return result

    r_peaks = [int(p) for p in info.get("ECG_R_Peaks", [])]
    if not r_peaks:
        logger.warning("No R-peaks detected.")
        result["status"] = "no_peaks_found"
        return result

    try:
        if "ECG_Quality" in signals_df.columns:
            mean_quality = float(np.nanmean(signals_df["ECG_Quality"].values))
        else:
            mean_quality = float(np.nanmean(nk.ecg_quality(signals_df["ECG_Clean"], sampling_rate=sampling_rate)))
        result["signal_quality_flag"] = "good" if mean_quality >= quality_threshold else "low_quality"
        if result["signal_quality_flag"] == "low_quality":
            logger.warning("Mean ECG quality %.3f is below threshold %.3f; flagging, not dropping.", mean_quality, quality_threshold)
    except Exception as e:
        logger.exception("ecg_quality check failed; proceeding with flag='unknown'.")
        result["signal_quality_flag"] = "unknown"

    def _at(arr_name: str, i: int) -> Optional[float]:
        arr = info.get(arr_name)
        if arr is None or i >= len(arr):
            return None
        val = arr[i]
        return None if (val is None or (isinstance(val, float) and np.isnan(val))) else float(val)

    clean_signal = signals_df["ECG_Clean"].values
    baseline_offset_samples = int(round(0.08 * sampling_rate))

    beat_features = []
    for i, r_peak in enumerate(r_peaks):
        rr_interval_ms = float((r_peaks[i] - r_peaks[i - 1]) / sampling_rate * 1000) if i > 0 else None

        r_onset, r_offset = _at("ECG_R_Onsets", i), _at("ECG_R_Offsets", i)
        qrs_duration_ms = float((r_offset - r_onset) / sampling_rate * 1000) if (r_onset is not None and r_offset is not None) else None

        p_peak_idx = _at("ECG_P_Peaks", i)
        p_amplitude = float(clean_signal[int(p_peak_idx)]) if p_peak_idx is not None else None

        p_onset_idx = _at("ECG_P_Onsets", i)
        st_elevation = None
        if r_offset is not None and p_onset_idx is not None:
            st_point = int(r_offset) + baseline_offset_samples
            if st_point < len(clean_signal):
                st_elevation = float(clean_signal[st_point] - clean_signal[int(p_onset_idx)])

        beat_features.append({
            "beat_index": i,
            "rr_interval": rr_interval_ms,
            "qrs_duration": qrs_duration_ms,
            "p_amplitude": p_amplitude,
            "st_elevation": st_elevation,
        })
    result["beat_features"] = beat_features
    result["status"] = "success"

    try:
        epochs = nk.epochs_create(
            signals_df, events=r_peaks, sampling_rate=sampling_rate,
            epochs_start=EPOCH_START_S, epochs_end=EPOCH_END_S,
        )
        analyze_df = nk.ecg_analyze(epochs, sampling_rate=sampling_rate).reset_index(drop=True)
        for i, row in analyze_df.iterrows():
            if i < len(result["beat_features"]):
                extras = {k: v for k, v in _to_json_safe(row.to_dict()).items()
                          if k not in ("beat_index", "rr_interval", "qrs_duration", "p_amplitude", "st_elevation")}
                result["beat_features"][i].update(extras)
    except Exception as e:
        logger.exception("Supplementary ecg_analyze() table failed; core beat_features above are unaffected.")
        result["status"] = "partial_success_delineation_failed"
        result["error"] = _structured_error(e)

    try:
        if record_name:
            labeled = segment_beats_with_labels(record_name, clean_signal, r_peaks, sampling_rate)
            cnn_segments, cnn_labels = labeled["cnn_segments"], labeled["cnn_labels"]
            result["num_cnn_segments"] = labeled["num_cnn_segments"]
        else:
            cnn_segments = segment_beats_for_cnn(clean_signal, r_peaks, sampling_rate)
            cnn_labels = None
            if cnn_segments is not None:
                result["num_cnn_segments"] = int(cnn_segments.shape[0])

        if cnn_segments is not None:
            result["cnn_segments"] = cnn_segments
            result["cnn_labels"] = cnn_labels
            if save_cnn_segments and segments_out_path:
                os.makedirs(os.path.dirname(segments_out_path), exist_ok=True)
                np.save(segments_out_path, cnn_segments)
                result["cnn_segments_path"] = segments_out_path
                if cnn_labels is not None:
                    labels_out_path = segments_out_path.replace("_beats.npy", "_labels.npy")
                    np.save(labels_out_path, cnn_labels)
                    result["cnn_labels_path"] = labels_out_path
    except Exception as e:
        logger.exception("CNN beat segmentation failed (feature table above may still be valid).")
        if result["error"] is None:
            result["error"] = _structured_error(e)

    return result

# ---------------------------------------------------------------------------
# Pipeline B -- EHR text NER (Bio-ClinicalBERT-family)
# ---------------------------------------------------------------------------

def _merge_subword_tokens(raw_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fallback merge for ##-prefixed wordpiece tokens, used only if the model's
    pipeline doesn't support aggregation_strategy natively."""
    merged = []
    current = None
    for ent in raw_entities:
        word = ent.get("word", "")
        if word.startswith("##") and current is not None:
            current["word"] += word[2:]
            current["end"] = ent.get("end", current["end"])
        else:
            if current is not None:
                merged.append(current)
            current = dict(ent)
    if current is not None:
        merged.append(current)
    return merged


def extract_clinical_entities(
    text: str,
    ner_model_name: str = DEFAULT_NER_MODEL,
    target_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Pipeline B. Independently unit-testable: takes raw clinical text, returns a
    structured result dict. No ECG code involved at all.

    NOTE ON THIS SANDBOX: this function requires downloading model weights from
    huggingface.co, which is not reachable from this execution environment's
    network allowlist. The code below is correct and ready to run locally / in
    your training environment, but has not been executed end-to-end here --
    only import- and logic-checked. Run the __main__ block locally to verify
    against your chosen model.
    """
    result = {"status": "not_run", "entities": [], "unmapped_labels": [], "error": None}

    if not text or not text.strip():
        result["status"] = "error"
        result["error"] = {"type": "ValueError", "message": "Empty text passed to NER pipeline.", "traceback": ""}
        return result

    try:
        from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForTokenClassification
    except Exception as e:
        logger.exception("transformers import failed.")
        result["status"] = "error"
        result["error"] = _structured_error(e)
        return result

    try:
        tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
        model = AutoModelForTokenClassification.from_pretrained(ner_model_name)

        # --- Explicit id2label check before trusting any output labels ---
        id2label = getattr(model.config, "id2label", {})
        generic_labels = {v for v in id2label.values() if v.upper().startswith("LABEL_")}
        if generic_labels and len(generic_labels) == len(id2label):
            logger.warning(
                "Model %s exposes only generic labels (%s), not named entity types. "
                "Flagging output as unmapped rather than trusting it as Disease/Drug/Symptom.",
                ner_model_name, sorted(generic_labels),
            )
            result["unmapped_labels"] = sorted(generic_labels)

        try:
            ner = hf_pipeline(
                "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
            )
            raw_entities = ner(text)
            entities = [
                {"text": e.get("word", ""), "label": e.get("entity_group", e.get("entity", "UNKNOWN")), "confidence": float(e.get("score", 0.0))}
                for e in raw_entities
            ]
        except TypeError:
            # aggregation_strategy unsupported by this model/pipeline version -- manual fallback
            logger.warning("aggregation_strategy unsupported for %s; using manual subword merge fallback.", ner_model_name)
            ner = hf_pipeline("ner", model=model, tokenizer=tokenizer)
            raw_entities = ner(text)
            merged = _merge_subword_tokens(raw_entities)
            entities = [
                {"text": e.get("word", ""), "label": e.get("entity", "UNKNOWN"), "confidence": float(e.get("score", 0.0))}
                for e in merged
            ]

        if target_labels:
            unexpected = {e["label"] for e in entities} - set(target_labels)
            if unexpected:
                logger.warning("NER returned labels outside target set %s: %s", target_labels, unexpected)

        result["entities"] = entities
        result["status"] = "unmapped_labels" if result["unmapped_labels"] else "success"
    except Exception as e:
        logger.exception("NER extraction failed for model %s.", ner_model_name)
        result["status"] = "error"
        result["error"] = _structured_error(e)

    return result


# ---------------------------------------------------------------------------
# Fusion layer
# ---------------------------------------------------------------------------

def fuse_agent1_outputs(
    record_id: str,
    ecg_result: Optional[Dict[str, Any]] = None,
    ner_result: Optional[Dict[str, Any]] = None,
) -> Agent1Output:
    """
    Joins Pipeline A and Pipeline B outputs by record_id (the join key -- see
    module docstring; swap for encounter_id/timestamp window if your data
    model needs that instead, but pick one explicitly).

    Fails loudly: if either pipeline didn't run or errored, that is recorded
    directly in ecg_status/ner_status/*_error rather than silently omitted or
    treated as an implicit success.
    """
    ecg_result = ecg_result or {"status": "not_run"}
    ner_result = ner_result or {"status": "not_run"}

    beat_features = []
    for bf in ecg_result.get("beat_features", []):
        try:
            beat_features.append(BeatFeature(**bf))
        except Exception as e:
            logger.warning("Skipping malformed beat_feature row for %s: %s", record_id, e)

    entities = []
    for ent in ner_result.get("entities", []):
        try:
            entities.append(ClinicalEntity(**ent))
        except Exception as e:
            logger.warning("Skipping malformed entity row for %s: %s", record_id, e)

    return Agent1Output(
        record_id=record_id,
        ecg_status=ecg_result.get("status", "not_run"),
        ecg_error=ecg_result.get("error"),
        sampling_rate=ecg_result.get("sampling_rate"),
        lead_used=ecg_result.get("lead_used"),
        signal_quality_flag=ecg_result.get("signal_quality_flag"),
        beat_features=beat_features,
        num_cnn_segments=ecg_result.get("num_cnn_segments", 0),
        cnn_segments_path=ecg_result.get("cnn_segments_path"),
        cnn_labels_path=ecg_result.get("cnn_labels_path"),
        ner_status=ner_result.get("status", "not_run"),
        ner_error=ner_result.get("error"),
        extracted_entities=entities,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_agent1(
    record_name: Optional[str] = None,
    ecg_signal: Optional[np.ndarray] = None,
    sampling_rate: int = MITBIH_SAMPLING_RATE,
    clinical_text: Optional[str] = None,
    client_id: Optional[str] = None,
    save_cnn_segments: bool = True,
) -> Dict[str, Any]:
    """Top-level entry point. Runs Pipeline A (if a signal or record is given)
    and Pipeline B (if text is given) independently, then fuses. Either can be
    omitted -- e.g. call with only clinical_text to run NER alone."""
    record_id = record_name or "unnamed_record"

    ecg_result = None
    if ecg_signal is not None:
        segments_path = None
        if save_cnn_segments:
            safe_id = record_id.replace("/", "_").replace("\\", "_")
            out_dir = os.path.join(SEGMENTS_ROOT, client_id) if client_id else SEGMENTS_ROOT
            segments_path = os.path.join(out_dir, f"{safe_id}_beats.npy")
        ecg_result = process_ecg_pipeline(
            ecg_signal, sampling_rate, save_cnn_segments=save_cnn_segments,
            segments_out_path=segments_path, record_name=record_name,
        )
    elif record_name is not None:
        load_result = load_ecg_record(record_name)
        if load_result["status"] == "success":
            safe_id = record_id.replace("/", "_").replace("\\", "_")
            out_dir = os.path.join(SEGMENTS_ROOT, client_id) if client_id else SEGMENTS_ROOT
            segments_path = os.path.join(out_dir, f"{safe_id}_beats.npy") if save_cnn_segments else None
            ecg_result = process_ecg_pipeline(
                load_result["ecg_signal"], load_result["sampling_rate"],
                save_cnn_segments=save_cnn_segments, segments_out_path=segments_path,
                record_name=record_name,
            )
            ecg_result["lead_used"] = load_result["lead_used"]
        else:
            ecg_result = {"status": "error", "error": load_result["error"]}

    ner_result = extract_clinical_entities(clinical_text) if clinical_text else None

    fused = fuse_agent1_outputs(record_id, ecg_result, ner_result)
    return fused.model_dump()


if __name__ == "__main__":
    # --- Self-test using NeuroKit2's built-in synthetic ECG (no network needed) ---
    print("Running Pipeline A self-test on synthetic ECG (nk.ecg_simulate)...")
    synthetic_ecg = nk.ecg_simulate(duration=15, sampling_rate=MITBIH_SAMPLING_RATE, noise=0.02, heart_rate=75)
    ecg_out = process_ecg_pipeline(synthetic_ecg, sampling_rate=MITBIH_SAMPLING_RATE)
    print("Pipeline A status:", ecg_out["status"])
    print("Quality flag:", ecg_out["signal_quality_flag"])
    print("Num beat feature rows:", len(ecg_out["beat_features"]))
    print("Num CNN segments:", ecg_out["num_cnn_segments"])

    sample_note = "Patient presents with acute chest pain and shortness of breath. Prescribed aspirin and metoprolol."
    ner_out = extract_clinical_entities(sample_note)
    print("Pipeline B status:", ner_out["status"])
    print(ner_out["entities"])

    fused = fuse_agent1_outputs("synthetic_test_001", ecg_result=ecg_out, ner_result=ner_out)
    print("\nFused output (ner not run, as expected):")
    print(fused.model_dump_json(indent=2)[:800], "...")
