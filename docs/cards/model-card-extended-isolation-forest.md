---
title: "Model Card — Extended Isolation-Forest (isotree, depth-aware)"
status: "production-candidate"
owners:
  - team: Lightning Sim / Detection
    email: john.goodacre@example.co.uk
model-id: "lightning_sim.detectors.ext_isoforest.ExtendedIsoForest"
classification: "Official Sensitive"
license: "Official Sensitive"
tags:
  - unsupervised
  - window-level
  - isolation-forest
  - isotree
  - feature-based
  - depth-threshold
  - reproducible
---

# Model Card — Extended Isolation-Forest (isotree, depth-aware)

> **TL;DR**  
> A per-station **Isolation-Forest** built on **isotree** with depth-aware scoring. Uses the 16-D **iso16** feature block, **RobustScaler**, a light **grid search** over contamination (via depth quantiles), and an **extreme-depth rescue**. Produces Boolean window masks plug-compatible with the shared evaluator.

---

## 1) Model summary

- **Task**: Unsupervised, window-level lightning-stroke detection (one model per *station*; network decision handled by evaluator).  
- **Signal → Features**: Raw int16 ADC (14-bit) → windows (**WIN=1024**, **HOP=512**) → **iso16** features (peak/median envelope, STA/LTA, crest factors, four FFT bands, wavelet ratio, spectral centroid/bandwidth/entropy).  
- **Backend**: :mod:`isotree` **IsolationForest** (C++), exposing **average depth** directly and scaling to hundreds of trees quickly.  
- **Pre-processing**: :class:`sklearn.preprocessing.RobustScaler` (median/IQR).  
- **Decision**: Windows with **depth < threshold** are anomalous; plus an **extreme-depth rescue** to catch ultra-rare events.  
- **Output**: `Dict[str, np.ndarray[bool]]` → station → hot windows.  
- **Evaluator**: `evaluate_windowed_model` (station and network metrics).

---

## 2) Intended use & scope

- **Primary**: Higher-recall, still-unsupervised detector that leverages richer features and a depth-aware threshold.  
- **Secondary**: Diversity partner for CDAE / OCSVM in OR/majority ensembles.  
- **Out-of-scope**: IC/CG typing, localisation, or safety-critical alerting without downstream checks.

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

Isolation-Forest isolates a point by random splits; **shorter paths** imply **more anomalous** inputs.  
For a subsample size \(n\), the expected path length of an unsuccessful search is

$$
c(n) \,=\, 2H(n-1) - \frac{2(n-1)}{n}, \qquad
H(m) \,=\, \sum_{k=1}^{m} \frac{1}{k} \;\approx\; \ln(m) + \gamma.
\tag{1}
$$

Let \(h(x)\) be the average **path depth** of sample \(x\) across trees.  
A common anomaly score is

$$
s(x) \,=\, 2^{-\, h(x) / c(n)}, \qquad \text{so } h \downarrow \;\Rightarrow\; s \uparrow.
\tag{2}
$$

**isotree** returns **average depth** (larger = more normal).  
We therefore declare an anomaly if

$$
\text{hot}(x) \,=\, \mathbb{1}\!\left[\, h(x) < T \;\lor\; h(x) < T_{\text{ext}} \,\right].
\tag{3}
$$

where:

- \(T\) is a **grid-tuned depth quantile** per station (e.g., first threshold whose mask keeps ≥ 0.1 % of windows), and  
- \(T_{\text{ext}}\) is the **extreme-depth rescue** (e.g., 0.05-percentile of the *training* depth distribution).

This produces a **recall-friendly** mask while keeping depth semantics explicit.

---

## 5) Algorithm overview

For each station:

1. **Windowing**: `win=1024`, `hop=512`.  
2. **iso16 features**: 16-D vector per window.  
3. **Robust scaling**: fit on station features; transform to `Xs`.  
4. **Train isotree IF**: e.g., `ntrees=200`, `ndim = d - 1`, `prob_pick_avg_gain=0`, `prob_pick_pooled_gain=0`, `sample_size='auto'`, `nthreads = (cpu_count-1)`.  
5. **Depth vector**: `depth = avg_depth(Xs)` (API accommodates `type="avg_depth"`/`output_type="avg_depth"` differences; fallback to `-predict`).  
6. **Grid search**: a tiny grid over contamination candidates (e.g., 0.1 % … 0.7 %); pick the first depth quantile whose mask keeps at least ~0.1 % windows.  
7. **Extreme rescue**: compute \(T_{\text{ext}}\) as the **lower \(100-\mathrm{EXTREME\_Q}\)** percentile of the *training* depth (e.g., EXTREME_Q=99.95 → 0.05-percentile).  
8. **Hot mask**: `depth < T or depth < T_ext`.

Deterministic with fixed `random_seed` and stable feature ordering.

---

## 6) Hyper-parameters (defaults)

- **Windowing**: `win=1024`, `hop=512`  
- **Features**: `FeatureExtractor(["iso16"])`  
- **Scaling**: `RobustScaler()`  
- **isotree**: `ntrees=200`, `sample_size='auto'`, `ndim=d-1`, `prob_pick_avg_gain=0`, `prob_pick_pooled_gain=0`, `nthreads=max(CPU-1,1)`, `random_seed=42`  
- **Grid**: contamination candidates `GRID_CONT = linspace(0.001, 0.007, 5)`  
- **Rescue**: `EXTREME_Q = 99.95` (rescues lowest 0.05 % depth)  

**Tuning tips**  
- Increase top-end of `GRID_CONT` for noisier stations; reduce for very clean stations.  
- `ndim ≈ d−1` improves tree diversity without over-fitting.  
- Keep gain heuristics off (`prob_pick_* = 0`) for truly random trees.

---

## 7) Usage (reference API)

**Class**: `lightning_sim.detectors.ext_isoforest.ExtendedIsoForest`

```python
from lightning_sim.detectors.ext_isoforest import ExtendedIsoForest

model = ExtendedIsoForest(
    win=1024, hop=512,
    base_cont=0.0015,
    n_trees=200,
    extreme_q=99.95
)

# Train one isotree forest per station
model.fit(storm_data.quantised, fs=109_375, verbose=True)

# Predict station hot masks
hot = model.predict(storm_data.quantised, fs=109_375)
```

**Interface**
- `__init__(win: int = 1024, hop: int = 512, base_cont: float = 0.0015, n_trees: int = 200, extreme_q: float = 99.95)`  
- `fit(raw: Dict[str, np.ndarray], fs: int, verbose: bool = True) -> None`  
- `predict(raw: Dict[str, np.ndarray], fs: int) -> Dict[str, np.ndarray]`

---

## 8) Evaluation protocol

- **Station-level** (window granularity): confusion matrix over windows.  
- **Network-level** (stroke granularity): stroke counted **TP** if ≥ `min_stn` stations are hot in any overlapping window; FP are **clusters** with no overlap to truth.

**Metrics**: Precision, Recall, F1 at **network** level; per-station P/R/F1 for diagnostics.

---

## 9) Expected behaviour

- **Higher recall** than simple thresholds in moderate/low SNR due to richer features and depth-based tuning.  
- **Stable latency** (no temporal smoothing inside the model; smoothing can be applied downstream if needed).  
- **Robust to outliers** via median/IQR scaling; depth distribution offers interpretable station diagnostics.

---

## 10) Risks, limitations & failure modes

- **Depth quantile mis-set**: too conservative → misses; too liberal → FP clusters. Keep the grid narrow and station-specific.  
- **Feature drift** (new RFI tones): depth distribution shifts → re-fit or update thresholds.  
- **API drift** in :mod:`isotree`: depth keyword changed across versions; wrapper `_avg_depth` handles `type` vs `output_type`; fall back flips sign of `predict` if needed.

---

## 11) Security & privacy (secure environment)

- **Data locality**: Training and inference are in-process; no network calls.  
- **PII** (*Personally Identifiable Information*): ADC signals contain none; scrub any auxiliary metadata upstream.  
- **Classification**: **Official Sensitive** — handle, store, and share in line with your organisation’s Official Sensitive procedures.  
- **Reproducibility**: Fix `random_seed`; persist scaler, thresholds and model per station for deterministic deployment.

---

## 12) Reproducibility & versioning

- **Config hash**: `{win, hop, features=iso16, n_trees, base_cont/grid, extreme_q, random_seed}` + code version.  
- **Artefacts to log**: per-station scaler params; chosen depth thresholds `{T, T_ext}`; contamination grid & selected point; model binary.  
- **CI**: Regression test on a frozen synthetic storm to ensure identical masks across refactors.

---

## 13) Dependencies & environment

- Python ≥ 3.9  
- NumPy, SciPy, scikit-learn (for scaler)  
- **isotree** ≥ 0.4 (IsolationForest with depth outputs)

**Resource footprint (indicative)**  
- Memory: O(n_win × 16) per station + model.  
- CPU: Training ~1 s per station for ~30k windows, `ntrees=200` on laptop-class CPUs; prediction is faster.

---

## 14) Governance

- **Owners**: Lightning Sim · Detection  
- **On-call**: john.goodacre@example.co.uk  
- **Escalation**: #sim-detection (internal)

---

## 15) Changelog

- **v0.1.0** — Initial depth-aware isotree model + card (iso16 features, RobustScaler, contamination grid, extreme-depth rescue).
