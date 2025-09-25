
# Burst Detection with Deep Learning & Compression

## Introduction & Motivation
**Burst Detection** is a research‑oriented time‑series project focused on detecting short “burst” events (simulated lightning strikes) within a noisy continuous signal. The repository provides everything you need to:

* **Generate realistic synthetic data** that mimics real bursts.
* **Train and evaluate** three complementary detection pipelines:
  1. **Compression‐based** Normalised Compression Distance (NCD) — zero‑training baseline.
  2. **Autoencoder anomaly detector** — unsupervised 1‑D U‑Net.
  3. **RawResNet1D classifier** — supervised deep learning.

The code is intentionally small and pedagogical, aimed at students or engineers who want to understand time‑series burst detection end‑to‑end.

---

## Project Layout

```text
├── scripts/                 # CLI entry‑points
│   ├── sim_make.py          # generate synthetic recording
│   ├── train_ae.py          # modern AE training (Lightning)
│   ├── train_ae_baseline.py # legacy AE training (raw PyTorch)
│   ├── train_resnet.py      # supervised ResNet training
│   ├── eval_ae.py           # AE burst detection
│   ├── eval_ae_baseline.py  # legacy AE evaluation
│   ├── eval_resnet.py       # ResNet evaluation
│   └── run_ncd.py           # NCD detector (no training)
├── leela_ml/                # core library code
│   ├── signal_sim/          # synthetic waveform simulator
│   ├── datamodules_npy.py   # StrikeDataset window loader
│   ├── models/              # neural network definitions
│   │   ├── dae_unet.py
│   │   ├── dae_unet_baseline.py
│   │   ├── raw_resnet.py
│   │   └── ncd.py
├── configs/                 # YAML hyper‑parameter files
├── data/                    # synthetic / real waveforms live here
├── reports/                 # metrics & plots land here
├── notebooks/               # interactive EDA demos
├── requirements.txt
└── README.md                # you are here
```

> **Dependencies:** Python ≥ 3.9, PyTorch 2 .x, PyTorch‑Lightning, NumPy, SciPy, scikit‑learn, matplotlib, seaborn.  
> GPU optional – runs on CPU albeit slower.

---

## 1 Generating Synthetic Data

The simulator creates a long noisy waveform with embedded burst events.

```bash
python scripts/sim_make.py     --minutes 5 \            # length of recording
    --out     data/storm5 \  # prefix for output files
    --seed    42             # RNG seed for repeatability
```

**Outputs**

| File | Purpose |
|------|---------|
| `storm5_<hash>.npy` | float32 waveform |
| `storm5_<hash>.json` | meta: sample‑rate + burst timestamps |
| `storm5_wave.npy` | *alias* copy of first channel for convenience |

The metadata lists burst start times; default burst length = **40 ms**. Noise floor includes pink‑noise + sensor white‑noise; bursts carry Gaussian envelopes plus harmonics. Drift and ADC clipping simulate real hardware quirks.

During training or evaluation the continuous waveform is cut into fixed-size windows. A window receives a **positive label** if *any* portion overlaps the 40 ms burst span; otherwise it is labelled 0. High overlap values therefore mark several consecutive windows around each strike.

---

## 2 Unsupervised Pipelines

### 2·1 Autoencoder (modern)

```bash
python scripts/train_ae.py   --npy    data/storm5_wave.npy   --meta   data/storm5_meta.json   --chunk  4096 --overlap 0.5   --bs     128 --epochs 20   --depth  4 --base 16   --device cuda   --ckpt   lightning_logs/ae_best.ckpt
```

| Flag | Meaning | Typical |
|------|---------|---------|
| `--chunk` | window length (samples) | 2 k – 8 k |
| `--overlap` | data augmentation | 0.5 |
| `--depth` | U‑Net down/up levels | 4 |
| `--base` | filters in first conv | 16 / 32 |
| `--noise_std` | add Gaussian noise | 0.05–0.1 |

**Evaluation**

```bash
python scripts/eval_ae.py   --npy data/storm5_wave.npy --meta data/storm5_meta.json   --ckpt lightning_logs/ae_best.ckpt   --chunk 512 --overlap 0.9   --mad_k 6 --win_ms 100 --fig_dark
```

Produces window‑ & event‑level metrics plus plots:

* `reports/ae_error_curve.png`
* `reports/ae_events.png`
* `reports/ae_event_timeline.png`

### 2·2 Compression (NCD)

```bash
python scripts/run_ncd.py   --npy data/storm5_wave.npy --meta data/storm5_meta.json   --chunk 512 --overlap 0.9   --codec zlib --mad_k 6 --per_win_norm
```

No training required. Flags bursts where NCD spikes above rolling median + k × MAD.

---

## 3 Supervised Pipeline (RawResNet1D)

### 3·1 Training

```bash
python scripts/train_resnet.py   --npy data/storm5_wave.npy --meta data/storm5_meta.json   --chunk 8192 --overlap 0.75   --bs 64 --epochs 40   --accelerator gpu --devices 1   --ckpt lightning_logs/raw_best.ckpt
```

*Event‑aware* split ensures windows from the same burst never leak across train/val/test.  
Class imbalance handled by `WeightedRandomSampler`.

### 3·2 Evaluation

```bash
python scripts/eval_resnet.py   --npy data/storm5_wave.npy --meta data/storm5_meta.json   --chunk 8192 --ckpt lightning_logs/raw_best.ckpt   --bs 512
```

Outputs VAL & TEST AUROC / F1 and saves `reports/resnet_val_test.png`.

---

## 4 Method Comparison

| Method | Training need | Typical Event‑F1 (synthetic) | Strengths | Weaknesses |
|--------|---------------|------------------------------|-----------|------------|
| **NCD** | none | 0.60–0.75 | zero setup, explainable | slower, many FP |
| **Autoencoder** | unsup. noise only | 0.80–0.90 | adapts, no labels | threshold tuning |
| **ResNet** | labelled bursts | 0.90–0.97 | highest accuracy | needs labels |

---

## 5 Running via Python API

```python
from leela_ml.models.dae_unet import UNet1D
from leela_ml.datamodules_npy import StrikeDataset
# lightweight wrapper avoids importing torch
from leela_ml.ncd import ncd_adjacent

ds = StrikeDataset("data/storm5_wave.npy", "data/storm5_meta.json",
                   chunk_size=512, overlap=0.9)
x, _ = ds[0]           # torch Tensor (1, 512)
model = UNet1D(depth=4, base=16).eval()
with torch.no_grad():
    recon = model(x.unsqueeze(0))
err = (recon - x).abs().mean()
print("reconstruction error:", err.item())
```

You can likewise call `ncd_adjacent(ds.windows, per_win_norm=True)` to obtain
an NCD score vector that is invariant to per-window scale.

---

## 6 Troubleshooting

| Issue | Fix |
|-------|-----|
| `CUDA device not found` | Install CPU‑only wheel: `pip install torch==<ver>+cpu` |
| Large checkpoint rejected by GitHub | `git lfs install && git lfs track "*.pt"` |
| Training slow | Use `--precision 16`, reduce `--depth`, smaller `--chunk` |
| `TypeError: ncd_adjacent()` unknown keyword | Import via `from leela_ml.ncd import ncd_adjacent` |

---

## 7 Contributing

1. Fork & clone.  
2. Create feature branch.  
3. Run `ruff` + `black .`.  
4. Add unit tests under `tests/`.  
5. PR with clear description.

---

## 8 Future Work

* Multichannel fusion (multiple sensors).  
* Streaming (real‑time) detection.  
* Variational / flow‑based models for richer probabilistic scoring.  
* Hyper‑parameter sweeps via Optuna.

---

# Leela-ML (demo scaffolding)

A modular playground for signal modeling and anomaly detection with **clean docs**.

[![Docs](https://img.shields.io/badge/docs-MkDocs%20Material-blue)](#) <!-- replace # with your Pages URL -->

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install mkdocs mkdocs-material mkdocstrings mkdocstrings-python griffe \
  mkdocs-gen-files mkdocs-section-index mkdocs-jupyter pymdown-extensions

python -m mkdocs serve



*Project created on macOS, validated on Ubuntu 20.04 with CUDA 11.8.  Feel free to raise issues or PRs!*  


