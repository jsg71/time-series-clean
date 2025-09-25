---
tags: [demo, anomaly-detection]
license: MIT
library_name: numpy
language: en
---

# Model Card — SimpleMADDetector (demo)

## Model Details
- **Task**: Unsupervised point anomaly detection (1D signals)
- **Algorithm**: Rolling Median Absolute Deviation (MAD), threshold at \( \operatorname{median} + k\,\mathrm{MAD} \)
- **Reference impl**: `leela_ml.demo_docs.SimpleMADDetector`

## Intended Use
- Quick baseline anomaly flagging for waveform segments (QA, smoke tests).
- **Out of scope**: safety-critical decisions, fine-grained event classes.

## Training / Tuning
- **Fit**: stores global median on a “mostly normal” slice.
- **Hyperparams**: window `w` (default 256), threshold `k` (default 4.0).

## Evaluation
- **Scores**: \( \frac{\lvert x - \operatorname{median}\rvert}{\mathrm{MAD}_w} \)
- **Metrics**: PR-AUC (imbalanced), ROC-AUC for reference
- **Ops point**: select `k` for FPR at 0.1–1%

## Risks & Limitations
- Mean shift → stale median; sensitive to window length.
- No localization beyond point score.

## How to use
```python

from leela_ml.demo_docs import DetectorConfig, SimpleMADDetector
m = SimpleMADDetector(DetectorConfig(window=25, k=3.5)).fit(train_signal)
scores = m.score(test_signal)
labels = m.predict(test_signal)
