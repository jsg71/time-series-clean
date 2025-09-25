---
title: "Model Card — Hilbert‑Envelope Threshold Detector"
status: "baseline"
owners:
  - team: Lightning Sim / Detection
    email: john.goodacre@example.co.uk
model-id: "lightning_sim.detectors.hilbert.HilbertThresholdDetector"
classification: "Official Sensitive"
license: "Official Sensitive"
tags:
  - unsupervised
  - window-level
  - signal-processing
  - hilbert-envelope
  - reproducible
---

# Model Card — Hilbert‑Envelope Threshold Detector

> **TL;DR**  
> A training‑free, per‑station baseline detector. It computes the **Hilbert envelope peak** per window and flags the top‑`p` percentile as *hot*. Network‑level decisions are derived via the shared evaluator (quorum over stations).

---

## 1) Model summary

- **Task**: Unsupervised, window‑level lightning‑stroke detection (per‑station).  
- **Signal**: Raw int16 ADC (14‑bit), windowed at **WIN=1024**, **HOP=512**, sampling **FS=109 375 Hz**.  
- **Method**: For each window, compute the **maximum envelope amplitude** using the analytic signal via the **Hilbert transform**; threshold by a **station‑specific percentile**.  
- **Output**: `Dict[str, np.ndarray[bool]]` → station → hot windows.  
- **Evaluator**: `evaluate_windowed_model` (station- and network‑level metrics).

---

## 2) Intended use & scope

- **Primary**: Establish a transparent, deterministic baseline; enable apples‑to‑apples comparison with feature‑ and learning‑based models (IsoForest, CDAE, Graph‑CDAE, OCSVM).  
- **Secondary**: Conservative OR‑ensemble vote to boost recall where SNR is moderate‑to‑high.  
- **Out‑of‑scope**: Stroke classification (IC/CG), localisation, or safety‑critical alerting without downstream checks.

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

Let \(x[n]\) be a real, discrete‑time waveform within a window of length \(W\).  
The **analytic signal** is

$$
z[n] \,=\, x[n] + j\,\hat{x}[n], \quad
\mathcal{F}\{\hat{x}\}(\omega) \,=\, -\,j\,\operatorname{sgn}(\omega)\,X(\omega),
\tag{1}
$$

where \(\hat{x}\) is the **Hilbert transform** of \(x\) (implemented via FFT in practice), and \(X(\omega)\) is the DFT of \(x\).  
The **envelope** is the magnitude

$$
e[n] \,=\, |z[n]| \,=\, \sqrt{x[n]^2 + \hat{x}[n]^2}. \tag{2}
$$

For window \(i\), define the **envelope peak**

$$
p_i \,=\, \max_{0\le n < W} e_i[n]. \tag{3}
$$

Let \(\mathcal{P}_p\) denote the \(p\)-th percentile over the set \(\{p_i\}\) for a given station. The **decision rule** is

$$
\text{hot}_i \,=\, \mathbb{1}\!\left[\, p_i > \mathcal{P}_p(\{p_i\})\,\right]. \tag{4}
$$

This is **scale‑aware** (monotone with amplitude) and **polarity‑agnostic** (envelope magnitude).

**Complexity**  
For \(n_{\text{win}}\) windows, naive complexity is \(\mathcal{O}(n_{\text{win}} W \log W)\) using FFT‑based Hilbert (SciPy), performed vectorised across windows.

---

## 5) Algorithm overview

1. **Windowing**: create stride‑based views, \(W=1024, H=512\).  
2. **Envelope peaks**: compute \(p_i\) via Hilbert transform (vectorised along axis).  
3. **Percentile threshold** per station: set \(T=\mathcal{P}_{p}\) (default \(p=99.9\)).  
4. **Hot mask**: mark windows with \(p_i > T\).  
5. **Network scoring**: pass masks to the evaluator (quorum across stations).

Deterministic; no training; no randomness during fit/predict.

---

## 6) Hyper‑parameters (defaults)

- `win=1024`, `hop=512` — window geometry (samples)  
- `pct_thresh=99.9` — percentile \(p\) for station‑specific threshold  
- **Evaluator knobs**: `min_stn=2`, `tol_win=0`, `burst_len=int(0.04*FS)`

**Tuning tips**  
- Lower `pct_thresh` (e.g., 99.5) for more recall; raise for more precision.  
- Keep `tol_win=0` for latency‑precise comparisons; set to `1` if models exhibit sub‑window jitter.

---

## 7) Usage (reference API)

**Class**: `lightning_sim.detectors.hilbert.HilbertThresholdDetector`

```python
from lightning_sim.detectors.hilbert import HilbertThresholdDetector

model = HilbertThresholdDetector(pct_thresh=99.9)  # only key knob
hot, n_win = model.fit_predict(
    quantised=storm_data.quantised,
    win=1024, hop=512, fs=109_375
)
```

**Interface**
- `__init__(pct_thresh: float = 99.9)`  
- `fit_predict(quantised: Dict[str, np.ndarray], *, win: int, hop: int, fs: int) -> (Dict[str, np.ndarray], int)`

---

## 8) Evaluation protocol

- **Station‑level** (window granularity): confusion matrix over windows.  
- **Network‑level** (stroke granularity): stroke is **TP** if ≥ `min_stn` stations are hot in any overlapping window; FP are **clusters** of hot windows with no truth overlap.

**Reported metrics**: Precision, Recall, F1 (network), plus station P/R/F1 for diagnostics.

---

## 9) Expected behaviour

- High **precision** at moderate‑to‑high SNR; recall degrades as SNR drops.  
- Robust to polarity flips and small phase shifts (envelope magnitude).  
- Sensitive to strong line‑hum or slowly varying gain if not pre‑filtered upstream.

---

## 10) Risks, limitations & failure modes

- **Percentile drift**: very burst‑heavy scenes can inflate the threshold (fewer windows flagged). Mitigate by capping \(p\) (e.g., 99.0) for such scenes.  
- **Gain/variance shifts**: different noise floors per station argue for **per‑station** thresholds (already used).  
- **No timing smoothing**: may flicker at window boundaries; consider evaluator `tol_win=1` if needed.

---

## 11) Security & privacy (secure environment)

- **Data locality**: Signal processing is in‑process; no external calls.  
- **PII** (*Personally Identifiable Information*): None expected in ADC streams; redact any auxiliary metadata upstream.  
- **Classification**: **Official Sensitive** — handle, store, and share in line with your organisation’s Official Sensitive procedures.  
- **Determinism**: No stochastic steps; identical inputs yield identical outputs.

---

## 12) Reproducibility & versioning

- **Config hash**: `{win, hop, FS, pct_thresh}` + code version.  
- **Artefacts to log**: per‑station thresholds and envelope‑peak histograms.  
- **CI**: Freeze a synthetic storm and assert identical masks across runs.

---

## 13) Dependencies & environment

- Python ≥ 3.9  
- NumPy, SciPy (`scipy.signal.hilbert`)  
- (Optional) The shared `FeatureExtractor(["hilbert_peak"])` block from `lightning_sim.features.basic`

**Resource footprint (indicative)**  
- Memory: O(n_win × S) for envelope peaks.  
- CPU: dominated by FFT‑based Hilbert; typically < real‑time on modern CPUs.

---

## 14) Governance

- **Owners**: Lightning Sim · Detection  
- **On‑call**: john.goodacre@example.co.uk  
- **Escalation**: #sim-detection (internal)

---

## 15) Changelog

- **v0.1.0** — Initial modular card and API reference.
