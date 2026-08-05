# Frontend API integration

`POST /api/v1/analyse` accepts a complete one-dimensional NumPy ECG recording
as multipart form data. The API performs segmentation internally and returns
recording-level predictions, ordered beat results, optional detailed
explanations, and an optional compact Grad-CAM overlay.

## Request fields

Required:

- `file`: the `.npy` ECG recording.
- `sampling_rate_hz`: the recording sampling rate.

Optional:

- `record_id` and `lead_name`.
- `include_explanations` (default `true`): include detailed per-beat maps.
- `include_overlay` (default `true`): include the compact recording overlay.
- `overlay_start_sample`: inclusive overlay window start.
- `overlay_stop_sample`: exclusive overlay window stop.
- `overlay_downsample_limit`: maximum number of returned overlay samples.

Window coordinates always refer to the original uploaded recording. Invalid or
empty windows return HTTP 422. Windowing never rebases or shifts sample indices.

## Waveform and attribution alignment

The API does not transform or return a second copy of the uploaded waveform.
The dashboard should retain the one-dimensional uploaded waveform and align it
using `recording_attribution_overlay.sample_indices`.

All compact overlay fields are parallel arrays. Element `i` of
`sample_indices`, `timestamps_seconds`, `maximum_attributions`,
`mean_attributions`, `coverage_counts`, and `contributing_beat_indices`
describes the same exact original ECG sample.

`maximum_attributions` is the default heatmap intensity. `mean_attributions`
describes the mean when overlapping explained beats cover the same source
sample. `coverage_counts` is the number of explained beat windows contributing
to that sample. Zero coverage means that no explanation was produced for the
sample; it does not mean the sample is clinically normal or unimportant.

When downsampling is requested, the selected interval is divided into
contiguous bins. The exact source sample with the largest maximum attribution
is retained from every bin. Ties retain the earliest sample. Returned sample
indices may therefore be non-uniform but remain exact source coordinates.

## Prediction and Grad-CAM interpretation

Beat results are ordered by their original R-peak sequence index. Each result
contains its AAMI label, confidence, R-peak sample and timestamp, and the exact
216-sample source window.

Supported AAMI classes are:

- `N`: normal and bundle branch block beats.
- `S`: supraventricular ectopic beats.
- `V`: ventricular ectopic beats.
- `F`: fusion beats.
- `Q`: unknown or unclassifiable beats.

Grad-CAM is calculated against the predicted-class logit and normalized
independently within each beat to the range 0–1. Values from different beats
are therefore useful for locating model-sensitive regions but are not
globally calibrated clinical severity scores. Grad-CAM does not establish
causality, diagnosis, or physiological importance and requires clinical
interpretation alongside the waveform and other evidence.

## External dashboard example

```python
from pathlib import Path

import httpx
import numpy as np

ecg_path = Path("record.npy")
waveform = np.load(ecg_path)

with ecg_path.open("rb") as stream:
    response = httpx.post(
        "http://backend.example/api/v1/analyse",
        files={"file": (ecg_path.name, stream, "application/octet-stream")},
        data={
            "sampling_rate_hz": "360",
            "include_explanations": "false",
            "include_overlay": "true",
            "overlay_downsample_limit": "5000",
        },
        timeout=120,
    )

response.raise_for_status()
analysis = response.json()
overlay = analysis["recording_attribution_overlay"]

indices = np.asarray(overlay["sample_indices"], dtype=int)
waveform_for_overlay = waveform[indices]
heat = np.asarray(overlay["maximum_attributions"], dtype=float)
assert np.allclose(
    overlay["timestamps_seconds"],
    indices / analysis["signal"]["sampling_rate_hz"],
)

abnormal = next(
    (
        beat
        for beat in analysis["recording_summary"]["beat_results"]
        if beat["prediction"]["predicted_label"] != "N"
    ),
    None,
)
if abnormal is not None:
    navigate_to_seconds = abnormal["r_peak_timestamp_seconds"]
```

For interactive scrolling, request the visible source interval with
`overlay_start_sample` and `overlay_stop_sample`. Use the returned start and
stop metadata to verify the response window before plotting.
