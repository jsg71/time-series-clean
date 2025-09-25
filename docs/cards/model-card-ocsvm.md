--
title: "Model Card — One‑Class SVM (per‑station)"
status: "production-candidate"
owners:
  - team: Lightning Sim / Detection
    email: john.goodacre@example.co.uk
model-id: "lightning_sim.detectors.ocsvm.ExtendedOCSVM"
classification: "Official Sensitive"
license: "Official Sensitive"
tags:
  - unsupervised
  - window-level
  - svm
  - rbf-kernel
  - feature-based
  - reproducible
---

# Model Card — One‑Class SVM (per‑station)

> **TL;DR**  
> A per‑station **One‑Class SVM (RBF kernel)** over the 16‑D **iso16** feature block. Scales features with **RobustScaler**, chooses **γ** via a fast **median‑heuristic**, performs a tiny **unsupervised grid** over \((\nu,\gamma)\) using a **stability split**, and adds an **extreme‑tail rescue** on the SVM scores. Outputs Boolean window masks plug‑compatible with the shared evaluator; optional 3‑tap majority smoothing reduces flicker.

---

## 1) Model summary

- **Task**: Unsupervised, window‑level lightning‑stroke detection (one model per *station*; network decision handled by the evaluator).  
- **Signal → Features**: Raw int16 ADC (14‑bit) → windows (**WIN=1024**, **HOP=512**) → **iso16** features (time/envelope/spectrum).  
- **Method**: :class:`sklearn.svm.OneClassSVM` with **RBF kernel** after :class:`sklearn.preprocessing.RobustScaler`.  
- **Decision**: A window is *hot* if SVM score `< 0` **or** below an **extreme‑tail** threshold from the training score distribution.  
- **Output**: `Dict[str, np.ndarray[bool]]` per station.  
- **Evaluator**: `evaluate_windowed_model` (station‑ and network‑level metrics).

---

## 2) Intended use & scope

- **Primary**: A stronger feature‑based unsupervised detector than simple thresholds, with principled kernel scaling via γ heuristic and light per‑station tuning.  
- **Secondary**: Diversity member in ensembles (e.g., OR with IF/CDAE/NCD).  
- **Out‑of‑scope**: Stroke typing, localisation, or safety‑critical alerting without downstream checks.

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

## 4) Mathematical formulation

### 4.1 One‑Class SVM (Schölkopf et al.)
We solve

$$
\min_{w,\,\rho,\,\xi}\; \tfrac{1}{2}\lVert w\rVert_2^2 + \frac{1}{\nu n}\sum_{i=1}^n \xi_i - \rho
\quad\text{s.t.}\quad
\langle w,\,\Phi(x_i)\rangle \ge \rho - \xi_i,\; \xi_i \ge 0,
\tag{1}
$$

with feature map \(\Phi\) and **RBF kernel**

$$
K(x,x') = \exp\!\big(-\,\gamma\,\lVert x-x'\rVert_2^2\big). \tag{2}
$$

The **decision function** at test point \(x\) is

$$
s(x) = \sum_{i\in \mathcal{SV}} \alpha_i\,K(x, x_i) - \rho, \quad
\text{predict normal if } s(x) > 0,\; \text{anomalous otherwise}. \tag{3}
$$

(Scikit‑learn returns larger scores for *more normal* windows.)

### 4.2 Median‑heuristic for γ
A common scale choice is

$$
\gamma \approx \frac{1}{\operatorname{median}\big\{\lVert x_i - x_j\rVert_2^2\big\}_{i\ne j}}. \tag{4}
$$

We approximate the median with **anchors**: pick \(k\) anchors \(A\), compute pairwise squared distances \(\{\lVert X - A\rVert^2\}\), and take the median over that \(O(nk)\) set (robust and fast).

### 4.3 Extreme‑tail rescue
Let \(s_i\) be SVM scores over all training windows. We define an additional threshold

$$
T_{\text{ext}} = \operatorname{Perc}_{100-\mathrm{EXTREME\_Q}}(\{s_i\}). \tag{5}
$$

Final decision per window: **hot** if \(s < 0\) **or** \(s < T_{\text{ext}}\).

---

## 5) Algorithm overview

For each station:

1. **Windowing**: build windows with `win=1024`, `hop=512`.  
2. **iso16 features** → 16‑D per window; **RobustScaler** → `Xs_all`.  
3. **Subsample** for SVM fitting: take up to `train_max` (e.g., 3 000) rows \(\to X_s\) for speed.  
4. **γ heuristic**: compute fast median of squared distances on `X_s` ⇒ base \(\gamma_0\); make a small grid `gam_grid = {0.5, 1.0, 2.0} × γ0`.  
5. **Unsupervised stability split**: split `X_s` into halves A/B. For each \((\nu,\gamma)\) in the tiny grid, fit on A, measure flagged fraction on B (and vice versa). Choose the pair minimising \(|p_B-\nu| + |p_A-\nu|\).  
6. **Fit final model** on all `X_s` with selected \((\nu,\gamma)\).  
7. **Scores on all windows**: `s = oc.decision_function(Xs_all)`; compute \(T_{\text{ext}}\).  
8. **Hot mask**: `mask = (s < 0) | (s < T_ext)`; optional 3‑tap majority smoothing reduces flicker.

Deterministic once RNG seeds are fixed for the subsampling/splits.

---

## 6) Hyper‑parameters (defaults)

- **Geometry**: `win=1024`, `hop=512`  
- **Features**: `FeatureExtractor(["iso16"])`  
- **Scaling**: `RobustScaler()`  
- **SVM grid**: `BASE_NU=0.0015`; `NU_GRID=[0.00075, 0.0015, 0.003]` (bounded to [1e‑4, 2e‑2])  
- **Gamma grid**: `GAM_FACT=[0.5, 1.0, 2.0] × γ0` (γ0 from median‑heuristic)  
- **Extreme‑tail**: `EXTREME_Q=99.95` (rescues lowest 0.05 % of scores)  
- **Training cap**: `TRAIN_MAX=3000` for speed  
- **Anchors**: `ANCHORS=64` for γ approximation  
- **Seeds**: `SEED=42` (NumPy RNG)

**Tuning tips**  
- Increase `BASE_NU` on noisy stations to gain recall; reduce to tighten precision.  
- Consider extending `GAM_FACT` to `{0.25, 0.5, 1.0, 2.0}` if the heuristic is unstable.  
- If scenes are very burst‑heavy, lower `EXTREME_Q` (e.g., 99.9) to avoid over‑rescuing.

---

## 7) Usage (reference API)

**Class**: `lightning_sim.detectors.ocsvm.ExtendedOCSVM`

```python
from lightning_sim.detectors.ocsvm import ExtendedOCSVM

model = ExtendedOCSVM(
    win=1024, hop=512,
    extreme_q=99.95,
    train_max=3000,
    nu_grid=[0.00075, 0.0015, 0.003],
    gam_factors=[0.5, 1.0, 2.0]
)

# Train per-station OCSVMs
model.fit(storm_data.quantised, fs=109_375, verbose=True)

# Predict per-station hot masks (with small majority smoothing)
hot = model.predict(storm_data.quantised, fs=109_375, smooth=True)
```

**Interface**
- `__init__(win:int=1024, hop:int=512, extreme_q:float=99.95, train_max:int=3000, nu_grid=..., gam_factors=...)`  
- `fit(raw: Dict[str, np.ndarray], fs: int, verbose: bool = True) -> None`  
- `predict(raw: Dict[str, np.ndarray], fs: int, smooth: bool = True) -> Dict[str, np.ndarray]`

---

## 8) Evaluation protocol

- **Station‑level** (window granularity): confusion matrix over windows.  
- **Network‑level** (stroke granularity): stroke counted **TP** if ≥ `min_stn` stations are hot in any overlapping window; FP are **clusters** with no overlap to truth.

**Metrics**: Precision, Recall, F1 at **network** level; per‑station P/R/F1 for diagnostics.

---

## 9) Expected behaviour

- Competitive recall vs. IF on shape‑rich iso16 features, especially when γ is well‑scaled.  
- Stable precision thanks to RobustScaler and the stability‑split grid.  
- Benefits from majority smoothing to reduce boundary flicker.

---

## 10) Risks, limitations & failure modes

- **Kernel scale mis‑set** (γ): too small ⇒ over‑smooth decision boundary, misses; too large ⇒ many FP. Median‑heuristic and grid mitigate but do not eliminate risk.  
- **ν mis‑set**: underestimates anomaly rate ⇒ misses; overestimates ⇒ FP clusters.  
- **Computational cost**: kernel methods scale super‑linearly; the `TRAIN_MAX` cap is essential for long storms.  
- **Drift**: strong spectrum/RFI changes shift the feature distribution ⇒ refresh scaling and re‑fit.

---

## 11) Security & privacy (secure environment)

- **Data locality**: All operations in‑process; no external services.  
- **PII** (*Personally Identifiable Information*): ADC traces contain no PII; scrub auxiliary metadata upstream.  
- **Classification**: **Official Sensitive** — handle, store, and share in line with your organisation’s Official Sensitive procedures.  
- **Reproducibility**: Fix seeds for subsampling/splits; persist scaler, (ν,γ), and thresholds per station.

---

## 12) Reproducibility & versioning

- **Config hash**: `{win, hop, features=iso16, nu_grid, gam_factors, extreme_q, train_max, anchors, seed}` + code version.  
- **Artefacts to log**: per‑station scaler params; selected (ν,γ); SVM support vector counts; extreme thresholds; score histograms.  
- **CI**: Regression test on a frozen storm with fixed seeds to ensure identical masks across refactors.

---

## 13) Dependencies & environment

- Python ≥ 3.9  
- NumPy, SciPy  
- scikit‑learn ≥ 1.1 (OneClassSVM), joblib (implicit)

**Resource footprint (indicative)**  
- Memory: O(min(TRAIN_MAX, n_win)²) kernel cache during fit + O(n_win) for scores.  
- CPU: A few seconds per station for ~3k samples; prediction is linear in n_win.

---

## 14) Governance

- **Owners**: Lightning Sim · Detection  
- **On‑call**: john.goodacre@example.co.uk  
- **Escalation**: #sim-detection (internal)

---

## 15) Changelog

- **v0.1.0** — Initial OCSVM model + card (iso16 features, RobustScaler, γ heuristic, stability grid, extreme‑tail rescue, evaluator integration).
