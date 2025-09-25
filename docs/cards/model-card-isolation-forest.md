---
title: "Model Card — Isolation‑Forest (scikit‑learn)"
status: "baseline"
owners:
  - team: Lightning Sim / Detection
    email: john.goodacre@example.co.uk
model-id: "lightning_sim.detectors.isoforest.IsoForestModel"
classification: "Official Sensitive"
license: "Official Sensitive"
tags:
  - unsupervised
  - window-level
  - isolation-forest
  - tree-ensemble
  - feature-based
  - reproducible
---

# Model Card — Isolation‑Forest (scikit‑learn)

> **TL;DR**  
> A per‑station **Isolation‑Forest** detector over the 16‑dimensional **iso16** feature block. Uses **RobustScaler** and a small, fixed **contamination** to produce Boolean window masks plug‑compatible with the shared evaluator.

---

## 1) Model summary

- **Task**: Unsupervised, window‑level lightning‑stroke detection (one model per *station*; network decision handled by evaluator).  
- **Signal → Features**: Raw int16 ADC traces (14‑bit) → windows (**WIN=1024**, **HOP=512**) → **iso16** features (time‑, envelope‑ and spectrum‑based).  
- **Method**: :class:`sklearn.ensemble.IsolationForest` trained per station after **RobustScaler** (median/IQR) normalisation.  
- **Output**: `Dict[str, np.ndarray[bool]]` → station → hot windows (`True` if predicted as anomaly).  
- **Evaluator**: `evaluate_windowed_model` (station and network metrics).

---

## 2) Intended use & scope

- **Primary**: Feature‑based baseline ML detector that is more expressive than simple thresholds yet inexpensive and label‑free.  
- **Secondary**: Diversity component in ensembles (e.g. OR with CDAE or NCD).  
- **Out‑of‑scope**: Source localisation, stroke typing (IC/CG), or safety‑critical alerting without downstream checks.

---

## 3) Inputs & outputs (I/O contract)

**Input**  
`raw: Dict[str, np.ndarray]` — equal‑length 1‑D int16 arrays (one per station).

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

## 4) Mathematical formulation (Isolation‑Forest)

An Isolation‑Forest isolates a point by recursively partitioning the space with random axis‑aligned splits. **Anomalies** are isolated in **fewer splits** (short paths).

- Let \(h(x)\) be the **path length** of sample \(x\) averaged over the forest.  
- For a subsample size \(n\), the expected path length of an unsuccessful search in a binary tree is

$$
c(n) \,=\, 2\,H(n-1) - \frac{2(n-1)}{n}, \quad
H(m) \,=\, \sum_{k=1}^{m} \frac{1}{k} \;\approx\; \ln(m) + \gamma,
\tag{1}
$$

with Euler–Mascheroni constant \(\gamma\).  

- A common anomaly **score** is

$$
s(x) \,=\, 2^{-\,\frac{\mathbb{E}[h(x)]}{c(n)}},
\tag{2}
$$

so **smaller** depths \(\Rightarrow\) **larger** scores (more anomalous).

**scikit‑learn mapping**: `predict` returns **-1** for anomalies, **+1** for normal; `decision_function` is **higher for more normal** points.

---

## 5) Algorithm overview

For each station:

1. **Windowing**: build windows with `win=1024`, `hop=512`.  
2. **iso16 features**: compute a 16‑D vector per window (envelope stats, band energies, wavelet ratio, spectral centroid/bandwidth/entropy).  
3. **Robust scaling**: fit `RobustScaler` on station features; transform.  
4. **Isolation‑Forest**: fit with `n_estimators = 150` and `contamination ≈ 0.0015` (tunable).  
5. **Hot mask**: `hot = (forest.predict(X_scaled) == -1)`.

Deterministic given fixed `random_state` and input ordering.

---

## 6) Hyper‑parameters (defaults)

- **Windowing**: `win=1024`, `hop=512`  
- **Features**: `FeatureExtractor(["iso16"])`  
- **Scaling**: `RobustScaler()`  
- **Forest**: `n_estimators=150`, `contamination=0.0015`, `random_state=42`  
- **Per‑station override**: optional `contam_map: Dict[str, float]` for stations with atypical noise floors

**Tuning tips**  
- Increase `contamination` to boost recall at the expense of precision.  
- Start with `n_estimators` in `100–200` for a good speed/variance trade‑off.  
- Keep `random_state` fixed for reproducibility in comparisons.

---

## 7) Usage (reference API)

**Class**: `lightning_sim.detectors.isoforest.IsoForestModel`

```python
from lightning_sim.features.basic import FeatureExtractor
from lightning_sim.detectors.isoforest import IsoForestModel

fx = FeatureExtractor(["iso16"])  # explicit injection (preferred)

model = IsoForestModel(
    fx=fx,
    win=1024, hop=512,
    contamination=0.0015,
    n_trees=150,
    random_state=42
)

# Train one forest per station
model.fit(storm_data.quantised, fs=109_375, verbose=True)

# Predict station hot masks
hot = model.predict(storm_data.quantised, fs=109_375)
```

**Interface**
- `__init__(fx, *, win: int = 1024, hop: int = 512, contamination: float = 0.0015, n_trees: int = 150, random_state: int = 42)`  
- `fit(raw: Dict[str, np.ndarray], fs: int, verbose: bool = True) -> None`  
- `predict(raw: Dict[str, np.ndarray], fs: int) -> Dict[str, np.ndarray]`

---

## 8) Evaluation protocol

- **Station‑level** (window granularity): confusion matrix over windows.  
- **Network‑level** (stroke granularity): stroke counted **TP** if ≥ `min_stn` stations are hot in any overlapping window; FP are **clusters** with no overlap to truth.

**Metrics**: Precision, Recall, F1 at **network** level; per‑station P/R/F1 for diagnostics.

---

## 9) Expected behaviour

- Improves recall over simple Hilbert thresholds in **moderate SNR** by mixing time/envelope/spectral cues.  
- May under‑perform on extreme low‑SNR or highly non‑stationary scenes unless `contamination` is tuned per station.  
- Stable across runs with fixed `random_state`.

---

## 10) Risks, limitations & failure modes

- **Contamination mis‑set**: too low → misses; too high → excess FP. Prefer per‑station overrides when noise floors differ.  
- **Feature drift**: abrupt spectrum changes (e.g., new RFI) can shift feature distributions → recalibration or re‑fit recommended.  
- **Axis‑aligned splits**: interactions captured only indirectly; deep trees help but increase variance.

---

## 11) Security & privacy (secure environment)

- **Data locality**: Training/inference are in‑process; no network calls.  
- **PII** (*Personally Identifiable Information*): ADC streams are expected to be PII‑free; ensure metadata redaction upstream.  
- **Classification**: **Official Sensitive** — handle, store, and share in line with your organisation’s Official Sensitive procedures.  
- **Reproducibility**: Fix `random_state`; persist scaler & model per station if you need bit‑identical deployment behaviour.

---

## 12) Reproducibility & versioning

- **Config hash**: `{win, hop, features=iso16, contamination, n_trees, random_state}` + code version.  
- **Artefacts to log**: per‑station scaler parameters; sklearn `estimators_` seeds; contamination overrides.  
- **CI**: Regression test on a frozen storm to ensure masks are unchanged across refactors.

---

## 13) Dependencies & environment

- Python ≥ 3.9  
- NumPy, SciPy  
- scikit‑learn ≥ 1.1 (IsolationForest), joblib (implicit in sklearn)

**Resource footprint (indicative)**  
- Memory: O(n_win × 16) per station (features) + model.  
- CPU: Fit typically sub‑second per station for ~30k windows, `n_trees=150` on a modern laptop CPU.

---

## 14) Governance

- **Owners**: Lightning Sim · Detection  
- **On‑call**: john.goodacre@example.co.uk  
- **Escalation**: #sim-detection (internal)

---

## 15) Changelog

- **v0.1.0** — Initial modular model + card (iso16 features, RobustScaler, per‑station forests).
