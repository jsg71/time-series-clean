---
title: "Model Card — Convolutional Denoising Auto-Encoder (CDAE)"
status: "production-candidate"
owners:
  - team: Lightning Sim / Detection
    email: john.goodacre@example.co.uk
model-id: "lightning_sim.detectors.cdae.CdaeModel"
classification: "Official Sensitive"
license: "Official Sensitive"
tags:
  - unsupervised
  - window-level
  - auto-encoder
  - convolutional
  - denoising
  - reproducible
---

# Model Card — Convolutional Denoising Auto-Encoder (CDAE)

> **TL;DR**  
> A per-station **Convolutional Denoising Auto-Encoder** trained to reconstruct *normal* background windows. At inference, windows with **high reconstruction error (MSE)** are flagged as anomalous. No labels required; outputs Boolean hot masks plug-compatible with the shared evaluator.

---

## 1) Model summary

- **Task**: Unsupervised, window-level lightning-stroke detection (one model per *station*; network decision handled by the evaluator).  
- **Signal**: Raw int16 ADC (14-bit), windowed at **WIN=1024**, **HOP=512**, sampling **FS=109 375 Hz**.  
- **Method**: Train a 1-D CNN **denoising** auto-encoder on normal windows (additive Gaussian noise at train time). Score windows by **per-window MSE** of reconstruction; threshold by a **station-specific percentile**.  
- **Output**: `Dict[str, np.ndarray[bool]]` → station → hot windows.  
- **Evaluator**: `evaluate_windowed_model` (station- and network-level metrics).

---

## 2) Intended use & scope

- **Primary**: Learned non-linear baseline that models station-specific background better than fixed statistics, remaining light enough for edge deployment.  
- **Secondary**: Complement to feature-based detectors (IsoForest/OCSVM) in OR/majority ensembles.  
- **Out-of-scope**: Source localisation, stroke typing (IC/CG), or safety-critical alerting without downstream checks.

---

## 3) Inputs & outputs (I/O contract)

**Input**  
`raw: Dict[str, np.ndarray]` — equal-length 1-D int16 arrays (one per station).

**Output**  
`hot: Dict[str, np.ndarray]` — Boolean masks of length `n_win` (aligned to the minimum window count across stations).

**Reference evaluator call**
```python
station_m, net_m, _ = evaluate_windowed_model(
    hot=hot,
    stroke_records=storm_data.stroke_records,
    quantized=storm_data.quantised,
    station_order=list(raw),
    cfg=EvalConfig(win=1024, hop=512, fs=109_375,
                   burst_len=int(0.04*109_375),
                   min_stn=2, tol_win=0),
    plot=True
)
```

---


## 4) Mathematical formulation

Let \(x_i \in \mathbb{R}^{W}\) be a normalised window (int16 → float32, divided by 32768, so \(x_i \in [-1, 1]\)).  
At training time we form a noisy input

$$
\tilde{x}_i = x_i + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma^2 I). \tag{1}
$$

The CDAE with parameters \(\theta\) defines a mapping  
\(f_\theta: \mathbb{R}^{W} \to \mathbb{R}^{W}\).  
The **denoising objective** is

$$
\mathcal{L}(\theta) \,=\, \frac{1}{N} \sum_{i=1}^N \frac{1}{W}
\bigl\| f_\theta(\tilde{x}_i) - x_i \bigr\|_2^2. \tag{2}
$$

At inference, the **per-window error** is

$$
e_i \,=\, \frac{1}{W} \bigl\| f_\theta(x_i) - x_i \bigr\|_2^2. \tag{3}
$$

With a station-specific percentile threshold  
\(T = \operatorname{Perc}_p(\{e_i\})\) (default \(p=99.9\)), the decision rule is

$$
\text{hot}_i \,=\, \mathbb{1}\!\left[\, e_i > T \,\right]. \tag{4}
$$


---

## 5) Architecture (default)

Per station, a small 1-D convolutional auto-encoder:

**Encoder**
- `Conv1d(1→8, k=7, s=2, p=3)` → ReLU → length 1024→512  
- `Conv1d(8→16, k=7, s=2, p=3)` → ReLU → length 512→256  
- `Conv1d(16→32, k=7, s=2, p=3)` → ReLU → length 256→128  
- `Flatten` → `Linear(32×128 → latent)` → ReLU

**Decoder**
- `Linear(latent → 32×128)` → reshape  
- `ConvTranspose1d(32→16, k=7, s=2, p=3, output_padding=1)` → ReLU → length 128→256  
- `ConvTranspose1d(16→8,  k=7, s=2, p=3, output_padding=1)` → ReLU → length 256→512  
- `ConvTranspose1d(8→1,   k=7, s=2, p=3, output_padding=1)` → length 512→1024

**Latent dimension**: `latent=32` (≈32:1 compression for a 1024-sample window).  
**Noise for denoising**: \(\sigma\) (`noise_std`) \(= 0.02\) by default.

---

## 6) Algorithm overview

For each station:

1. **Windowing**: build stride-views with `win=1024`, `hop=512`.  
2. **Normalisation**: int16 → float32, divide by 32768 to \([-1, 1]\).  
3. **Training subset**: sample up to `train_win` (e.g., 20 000) windows for training.  
4. **Train CDAE**: add Gaussian noise (std `noise_std`), minimise MSE for `epochs` with `batch` size.  
5. **Score windows**: compute \(e_i\) for *all* windows (batched, e.g., 4096 at a time).  
6. **Threshold**: set \(T=\operatorname{Perc}_p(e)\) (default `p=99.9`).  
7. **Hot mask**: `hot = (e > T)`.

Determinism requires fixing RNG seeds (NumPy/Torch) and disabling nondeterministic CuDNN kernels where needed.

---

## 7) Hyper-parameters (defaults)

- **Geometry**: `win=1024`, `hop=512`  
- **Architecture**: `latent=32`  
- **Training**: `epochs=4`, `batch=256`, `train_win=20_000`, `noise_std=0.02`, `lr=1e-3 (Adam)`  
- **Thresholding**: `pct_thr=99.9` (per-station)  
- **Device**: auto (`cuda` → `mps` → `cpu`)

**Tuning tips**  
- Lower `pct_thr` (e.g., 99.5) to increase recall at modest precision cost.  
- Increase `latent` to reduce error variance (risk: over-fitting); decrease to emphasise anomalies.  
- `noise_std` in 0.01–0.05 works well; too high blurs structure, too low encourages memorisation.

---

## 8) Usage (reference API)

**Class**: `lightning_sim.detectors.cdae.CdaeModel`

```python
from lightning_sim.detectors.cdae import CdaeModel

model = CdaeModel(
    win=1024, hop=512,
    latent=32, epochs=4, batch=256,
    train_win=20_000, pct_thr=99.9
)

# Train one CDAE per station
model.fit(storm_data.quantised)

# Predict per-station hot masks
hot = model.predict(storm_data.quantised)
```

**Interface**
- `__init__(*, win: int = 1024, hop: int = 512, latent: int = 32, epochs: int = 4, batch: int = 256, train_win: int = 20000, pct_thr: float = 99.9, device: Optional[str] = None)`  
- `fit(raw: Dict[str, np.ndarray], verbose: bool = True) -> None`  
- `predict(raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]`

---

## 9) Evaluation protocol

- **Station-level** (window granularity): confusion matrix over windows.  
- **Network-level** (stroke granularity): stroke counted **TP** if ≥ `min_stn` stations are hot in any overlapping window; FP are **clusters** with no overlap to truth.

**Metrics**: Precision, Recall, F1 at **network** level; per-station P/R/F1 for diagnostics.

---

## 10) Expected behaviour

- Strong recall on mid-amplitude events; catches **shape changes** that feature baselines may miss.  
- May miss extreme spikes if activations saturate; complementary to Hilbert/IF detectors.  
- Benefits from modest training (few epochs) and per-station thresholds.

---

## 11) Risks, limitations & failure modes

- **Non-determinism** without fixed seeds; set seeds (NumPy/Torch) for auditability.  
- **Drift**: covariance changes (RFI, gain) require periodic refresh (few epochs).  
- **Threshold coupling**: burst-heavy scenes inflate `pct_thr` (fewer windows flagged); consider capping or using a mixture model on \(e\).  
- **Compute**: per-station training is GPU-friendly but still costs minutes for very long storms.

---

## 12) Security & privacy (secure environment)

- **Data locality**: All training/inference is in-process; no external calls.  
- **PII** (*Personally Identifiable Information*): none expected in ADC; scrub auxiliary metadata upstream.  
- **Classification**: **Official Sensitive** — handle, store, and share in line with your organisation’s Official Sensitive procedures.  
- **Model storage**: Persist per-station weights only within the secure perimeter; avoid leaks via logs.

---

## 13) Reproducibility & versioning

- **Config hash**: `{win, hop, latent, epochs, batch, train_win, pct_thr, noise_std, lr}` + code version.  
- **Artefacts to log**: per-station seeds, training loss curves, percentile thresholds, and scoring histograms.  
- **CI**: With fixed seeds and a frozen storm, assert identical masks across runs (allowing for floating-point tolerances).

---

## 14) Dependencies & environment

- Python ≥ 3.9  
- NumPy, PyTorch ≥ 1.13 (Conv1d/ConvTranspose1d)  
- (Optional) CUDA or Metal (MPS) for acceleration

**Resource footprint (indicative)**  
- **Memory**: O(batch × channels × 1024) activations per station during training.  
- **Throughput**: On laptop GPU, `epochs=4`, `batch=256`, `train_win=20k` typically < 1–2 minutes per station; CPU is slower.

---

## 15) Governance

- **Owners**: Lightning Sim · Detection  
- **On-call**: john.goodacre@example.co.uk  
- **Escalation**: #sim-detection (internal)

---

## 16) Changelog

- **v0.1.0** — Initial CDAE model + card (denoising AE, percentile threshold, evaluator integration).
