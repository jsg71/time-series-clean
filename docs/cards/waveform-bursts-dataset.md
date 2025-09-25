---
tags: [demo, waveform, anomalies]
license: CC-BY-4.0
language: en
---

# Data Card — Waveform Bursts (demo)

## Dataset Details
- **Modality**: 1D time-series (float32)
- **Sampling rate**: `fs` Hz (fill in)
- **Provenance**: *(synthetic / device logs / simulator — describe)*
- **Size**: *(e.g., 100k samples; ~1k labeled bursts)*
- **Access**: *(link or internal path)*

## Composition
- **Unit of analysis**: sample or fixed-length window
- **Features**: raw amplitude (engineered features optional downstream)
- **Labels**: burst / not-burst (sample- or window-level)
- **Class balance**: highly imbalanced (rare bursts)

## Collection & Preprocessing
- **Collection**: *(device pipeline / simulator version / date range)*
- **Preprocessing**: detrend → normalize (per segment) → window(size, hop)
- **Filtering**: drop saturated segments; cap extreme spikes (optional)
- **Splits**:
  - **Train**: normal-only
  - **Val**: mostly normal, a few labeled bursts for thresholding
  - **Test**: mixed normal/anomalous

## Recommended Uses
- Unsupervised AD baselines (MAD, AE, OCSVM)
- Pipeline sanity checks and downstream feature evaluations

## Sensitive Attributes / Ethics
- Synthetic: generally N/A.
- Real data: document any PII removal, device IDs hashing, and consent/retention policy.

## Limitations & Caveats
- Synthetic bursts may not capture real failure modes.
- Stationarity assumptions may break across devices/sites.
- Label noise likely—treat metrics with caution.

## Schema (example)
| Field        | Type      | Description                          |
|--------------|-----------|--------------------------------------|
| `signal`     | float32[] | Raw samples, length `T`              |
| `fs`         | float32   | Sampling rate (Hz)                   |
| `label`      | int8      | 0/1 at sample or window level        |
| `segment_id` | str       | Source segment identifier            |
| `meta`       | json      | Optional metadata (device/site/etc.) |

## Usage Example
```python
import numpy as np
# replace with your actual loader path
from leela_ml.data import load_waveform_bursts  # ← your module
X_train, y_train, meta = load_waveform_bursts(split="train", window=512, hop=128)
```

