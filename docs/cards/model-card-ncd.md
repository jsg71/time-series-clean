---
title: "Model Card — NCDDetector (Normalised Compression Distance)"
status: "experimental"
owners:
  - team: Lightning Sim / Detection
    email: john.goodacre@example.co.uk
model-id: "lightning_sim.detectors.ncd.NCDModel"
classification: "Official Sensitive"
license: "Official Sensitive"
tags:
  - unsupervised
  - anomaly-detection
  - compression
  - window-level
  - reproducible
---

# Model Card — NCDDetector (Normalised Compression Distance)

> **TL;DR**  
> A model‑free, per‑station detector that flags windows whose byte‑encoded waveform fails to compress well with a station‑specific **baseline** window. Supports four encodings (`"bits"`, `"raw"`, `"norm"`, `"tanh"`), uses **bzip2** size as a surrogate distance, and produces **Boolean hot masks** compatible with the shared evaluator.

---

## 1) Model summary

- **Task**: Unsupervised, window‑level lightning stroke detection (one model per *station*, network decision done by evaluator).  
- **Signal**: Raw ADC int16 traces (14‑bit), windowed (default **WIN=1024**, **HOP=512**) at **FS=109 375 Hz**.  
- **Method**: **Normalised Compression Distance (NCD)** between each window and a **baseline** window selected from the most compressible windows (quiet background). Compression with **bzip2, level 9**.  
- **Outputs**: `Dict[str, np.ndarray[bool]]` → station → hot windows.  
- **Evaluator**: Use `evaluate_windowed_model` for station‑ and network‑level metrics (quorum, tolerance, FP clustering).

---

## 2) Intended use & scope

- **Primary**: Fast, training‑free baseline to sanity‑check more complex models (IF, CDAE, GNN‑CDAE) and to provide a robust non‑parametric detector when labels are scarce.  
- **Out‑of‑scope**: Fine‑grained stroke typing (IC vs CG), localisation, or safety‑critical decisions without downstream verification.

---

## 3) Algorithm (at a glance)

1. **Window view** (stride trick): produce matrix `W ∈ ℤ^{n_win×WIN}` per station.  
2. **Encode** each window → `bytes` using one of:  
   - `"bits"`: sign of first differences, packed (8/byte) — *shape‑only*.  
   - `"raw"`: verbatim int16.  
   - `"norm"`: z‑score → re‑quantise to int16.  
   - `"tanh"`: soft‑clip via `tanh(x/16384)` → int16.  
3. **Compressed sizes**: cache `C(w)` for all windows; select **baseline** as the median within the lowest `BASE_PCT` % by `C`.  
4. **NCD to baseline** for each window `w`:  

$$
\text{NCD}(w, b) = \frac{C(w+b) - \min(C(w), C(b))}{\max(C(w), C(b))}.
$$

5. **Adaptive threshold** per station: hot if `NCD > min(percentile, μ + Z·σ)`.  
6. **Output** per station: Boolean mask `hot[nm]` of length `n_win`.  

**Determinism**: Given the raw input and encoding, baseline selection and thresholds are deterministic (no stochastic fitting).

---

## 4) Encodings & trade‑offs

| Encoding | What it preserves | Why it might help | Weaknesses |
|---|---|---|---|
| `"bits"` | Sign of first differences (zero‑crossings, coarse shape) | Extremely compressible; spikes stand out | Amplitude‑blind; may miss quiet strokes |
| `"raw"`  | Full 16‑bit samples | No preprocessing assumptions | Gaussian noise & gain drift dominate size; FPs |
| `"norm"` | Z‑scored, re‑quantised | Equalises gain; focuses on shape | Over‑whitening under variance shift |
| `"tanh"` | Soft‑clipped mid‑range | Robust to saturation | May erase tiny precursors |

---

## 5) Hyper‑parameters

- **Windowing**: `win` (1024), `hop` (512)  
- **Baseline pool**: `base_pct` (default **5** %)  
- **Thresholding**: `pct_thr` (e.g. **98.5**), `z_sigma` (e.g. **3.5**)  
- **Encoding**: one of `{"bits","raw","norm","tanh"}`

**Tuning**: Increase `base_pct` if scenes are volatile; reduce `pct_thr` to gain recall; keep `z_sigma ≥ 3` for stability.

---

## 6) I/O contract

- **Input**: `raw: Dict[str, np.ndarray]` of equal‑length int16 arrays (one per station).  
- **Output**: `hot: Dict[str, np.ndarray[bool]]`, aligned to the minimum window count.  

**Evaluator** (unchanged):
```python
station_m, net_m, _ = evaluate_windowed_model(
    hot=hot,
    stroke_records=storm_data.stroke_records,
    quantized=storm_data.quantised,
    station_order=list(raw),
    cfg=EvalConfig(win=win, hop=hop, fs=FS, burst_len=int(0.04*FS),
                   min_stn=2, tol_win=0),
    plot=True
)
```

---

## 7) Usage (reference API)

The modular class is `NCDModel`:

```python
from lightning_sim.detectors.ncd import NCDModel

model = NCDModel(
    encoding="bits",   # "bits" | "raw" | "norm" | "tanh"
    win=1024,
    hop=512,
    base_pct=5.0,
    pct_thr=98.5,
    z_sigma=3.5
)

# Fit (no labels) and predict
model.fit(storm_data.quantised, verbose=True)
hot = model.predict(storm_data.quantised)
```

**Interface summary**
- `__init__(encoding, win, hop, base_pct, pct_thr, z_sigma)`  
- `fit(raw: Dict[str, np.ndarray], verbose: bool = True) -> None`  
- `predict(raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]`

---

## 8) Evaluation protocol

- **Station‑level** (window granularity): confusion matrix over windows.  
- **Network‑level** (stroke granularity): a stroke is **TP** if ≥ `min_stn` distinct stations are hot in any overlapping window; FP are **clusters** of hot windows with no truth overlap.

Metrics: Precision, Recall, F1 (network). Station P/R/F1 for diagnostics.

---

## 9) Expected behaviour

- `"bits"` often strongest baseline F1 at moderate SNR (reacts to structure).  
- `"raw"` tends to over‑flag in noisy or drifting conditions.  
- `"norm"`/`"tanh"` trade between amplitude sensitivity and robustness.

---

## 10) Risks, limits & failure modes

- **Amplitude blindness** (`"bits"`) → misses low‑SNR subtle events.  
- **Noise‑dominated compression** (`"raw"`) → FPs from variance shifts or hum.  
- **Baseline brittleness** if `base_pct` too small.  
- **CPU cost**: compression dominates runtime; memoisation mitigates but large storms are slow.

---

## 11) Security, privacy & compliance

- **Data locality**: All work is in‑process; no external calls.  
- **PII** (*Personally Identifiable Information*): None expected in ADC; ensure metadata redaction upstream.  
- **Classification**: **Official Sensitive** — handle, store, and share in line with your organisation’s Official Sensitive procedures.  
- **Determinism**: No random components in inference; “fit” is deterministic given inputs.  
- **Supply chain**: Uses stdlib `bz2`; no shelling out.

---

## 12) Reproducibility & versioning

- **Config hash**: hash of `{encoding, win, hop, base_pct, pct_thr, z_sigma}` + code version.  
- **Artefacts to log**: baseline window indices, thresholds, optional NCD vectors.  
- **CI**: Smoke test that `predict()` is identical for a frozen storm.

---

## 13) Dependencies

- Python ≥ 3.9  
- NumPy; tqdm (optional)  
- Standard library **bz2**

---

## 14) Resource footprint (indicative)

- **Memory**: O(n_win) for sizes + NCD vector per station.  
- **CPU**: Compression is CPU‑bound; ~1–3× real‑time per station on a laptop CPU.

---

## 15) Changelog

- **v0.1.0** — Initial modular release: four encodings; adaptive threshold; evaluator integration.

---

## 16) References

- Cilibrasi & Vitányi (2005) — *Clustering by Compression* (NCD).
