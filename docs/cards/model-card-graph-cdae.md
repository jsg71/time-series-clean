---
title: "Model Card — Graph‑CDAE (GAT‑conditioned Denoising Auto‑Encoder)"
status: "experimental"
owners:
  - team: Lightning Sim / Detection
    email: john.goodacre@example.co.uk
model-id: "lightning_sim.detectors.graph_cdae.GraphCDaeModel"
classification: "Official Sensitive"
license: "Official Sensitive"
tags:
  - unsupervised
  - window-level
  - auto-encoder
  - graph-neural-networks
  - GATv2
  - denoising
  - reproducible
---

# Model Card — Graph‑CDAE (GAT‑conditioned Denoising Auto‑Encoder)

> **TL;DR**  
> A **multi‑station** denoising auto‑encoder that conditions each station’s latent code on a fully‑connected **graph** via **GATv2** attention blocks. Windows with high **reconstruction error** are flagged as anomalies, optionally gated by **Hilbert‑envelope peaks** and smoothed by a short majority filter. Outputs per‑station Boolean masks plug‑compatible with the shared evaluator.

---

## 1) Model summary

- **Task**: Unsupervised, window‑level lightning‑stroke detection with **cross‑station context**.  
- **Signal**: Raw int16 ADC (14‑bit), windowed at **WIN=1024**, **HOP=512**, sampling **FS=109 375 Hz**.  
- **Method**: Per‑window tensor \(X \in \mathbb{R}^{S\times W}\) (S stations). A 1‑D CNN **encoder** maps each station to a latent \(z_i \in \mathbb{R}^{L}\). Two **GATv2** graph blocks exchange information across stations: \(z'_i = \mathrm{GAT}(z, G)\). A mirrored **decoder** reconstructs \(\hat{X}\). Windows are scored by per‑station **reconstruction error**, then **robust‑z** and **envelope** gates produce the final hot mask.  
- **Output**: `Dict[str, np.ndarray[bool]]` → station → hot windows.  
- **Evaluator**: `evaluate_windowed_model` (station‑ and network‑level metrics).

---

## 2) Intended use & scope

- **Primary**: Capture **network‑level correlations** (e.g., coincident bursts, propagation similarities) that single‑station CDAE or feature models miss.  
- **Secondary**: A diversity component in ensembles with IF/OCSVM/NCD.  
- **Out‑of‑scope**: Source localisation or stroke typing; the model targets **detection** only.

---

## 3) Inputs & outputs (I/O contract)

**Input**  
`raw: Dict[str, np.ndarray]` — equal‑length 1‑D int16 arrays (one per station).

**Output**  
`hot: Dict[str, np.ndarray]` — Boolean masks of length `n_win` per station (min‑aligned window count).

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

### 4.1 Denoising objective
For window \(X \in \mathbb{R}^{S\times W}\) normalised to \([-1,1]\) and i.i.d. Gaussian noise \(\varepsilon \sim \mathcal{N}(0, \sigma^2)\),

$$
\tilde{X} = X + \varepsilon, \qquad
\mathcal{L}(\theta) = \frac{1}{S W}\,\| f_\theta(\tilde{X}, G) - X \|_F^2. \tag{1}
$$

### 4.2 GATv2 message passing (per block)
Let \(h_i = W z_i\) be a linear projection of the station latent. Attention coefficients are

$$
e_{ij} = a^\top \sigma( [h_i \Vert h_j] ), \qquad
\alpha_{ij} = \operatorname{softmax}_j(e_{ij}), \tag{2}
$$

and the updated latent is

$$
z_i^{(\mathrm{new})} = \mathrm{LN}\big( z_i + \sigma\!\left( \textstyle\sum_{j\in\mathcal{N}(i)} \alpha_{ij} h_j \right) \big), \tag{3}
$$

with LayerNorm (LN) and ReLU \(\sigma\). We use **GATv2Conv** with 4 heads (concat=False) in two residual blocks.

### 4.3 Scoring and gates
Per station \(i\) and window index \(t\), define the **MAE** reconstruction error

$$
r_{i,t} = \frac{1}{W}\sum_{n=1}^{W} \big\lvert \hat{x}_{i,t}[n] - x_{i,t}[n] \big\rvert. \tag{4}
$$

Compute robust z‑scores using median and MAD:

$$
m_i = \operatorname{median}_t\, r_{i,t},\quad
d_i = \operatorname{median}_t\, |r_{i,t}-m_i| + \varepsilon,\quad
z_{i,t} = \frac{r_{i,t}-m_i}{d_i}. \tag{5}
$$

Let \(p_{i,t}\) be the **Hilbert‑envelope peak** in window \(t\) at station \(i\).  
With envelope gate \(T_{\mathrm{env}}\) (e.g., 95‑th percentile) and z‑gate \(\tau\) (e.g., 2.5), the final decision is

$$
\text{hot}_{i,t} = \mathbb{1}\!\left[\, z_{i,t} > \tau \;\wedge\; p_{i,t} > T_{\mathrm{env}}\,\right], \tag{6}
$$

optionally **smoothed** by a majority filter (convolution with \([1,1,1]\) ≥ 2).

---

## 5) Architecture (default)

**Encoder (per station)**  
`Conv1d(1→8,k=7,s=2,p=3)` → ReLU → `Conv1d(8→16,7,2,3)` → ReLU → `Conv1d(16→32,7,2,3)` → ReLU → `Flatten` → `Linear(32×128→LAT)` → ReLU

**Graph blocks**  
Two residual **_GBlock**(LAT) modules: each uses `GATv2Conv(LAT→LAT, heads=4, concat=False)` + residual + LayerNorm.

**Decoder (per station)**  
`Linear(LAT→32×128)` → reshape → `ConvT1d(32→16,7,2,3,op=1)` → ReLU → `ConvT1d(16→8,7,2,3,op=1)` → ReLU → `ConvT1d(8→1,7,2,3,op=1)`

**Latent size**: `LAT=32` by default.

---

## 6) Algorithm overview

1. **Stack windows**: `RAW = stack([make_windows(raw[nm]) for nm in STN], axis=1) → (n_win, S, W)`; normalise to \([-1,1]\).  
2. **Train** (`epochs≈3`, `batch=256`): noisy input \(\tilde{X}\) → model → minimise MSE to reconstruct \(X\).  
3. **Score**: compute per‑station MAE \(r_{i,t}\) in batches.  
4. **Gates**: robust‑z (\(\tau=2.5\)) AND envelope‑peak (>95‑th percentile).  
5. **Smooth**: majority filter with `[1,1,1]` (optional).  
6. **Output**: Boolean masks `hot[nm]` for all stations.

---

## 7) Hyper‑parameters (defaults)

- **Geometry**: `win=1024`, `hop=512`  
- **Model**: `latent=32`, 2× GATv2 blocks (heads=4, concat=False)  
- **Training**: `epochs=3`, `batch=256`, `noise_std=0.03`, `lr=2e-3 (AdamW)`  
- **Gates**: `z_mad=2.5`, `env_pct=95`  
- **Smoothing**: majority kernel `[1,1,1]` (≥2)  
- **Graph**: fully‑connected directed edges without self‑loops

**Tuning tips**  
- Raise `env_pct` to tighten precision; lower `z_mad` to increase recall.  
- Reduce heads or latent for lighter edge deployment; increase for capacity.  
- Consider sparse graphs (nearest‑neighbour by geography) as S grows.

---

## 8) Usage (reference API)

**Class**: `lightning_sim.detectors.graph_cdae.GraphCDaeModel`

```python
from lightning_sim.detectors.graph_cdae import GraphCDaeModel, build_fully_connected_edges

S = len(storm_data.quantised)
edge_index = build_fully_connected_edges(S)  # shape = (2, S*(S-1))

model = GraphCDaeModel(
    win=1024, hop=512,
    latent=32, epochs=3, batch=256,
    noise_std=0.03, z_mad=2.5, env_pct=95,
    smooth_kernel=[1,1,1], device="cuda"  # or "mps"/"cpu"
)

# Train once on the stacked multi-station windows
model.fit(storm_data.quantised, edge_index=edge_index)

# Predict per-station hot masks
hot = model.predict(storm_data.quantised, edge_index=edge_index)
```

**Interface**
- `__init__(*, win:int=1024, hop:int=512, latent:int=32, epochs:int=3, batch:int=256, noise_std:float=0.03, z_mad:float=2.5, env_pct:float=95.0, smooth_kernel: list[int] = [1,1,1], device: Optional[str]=None)`  
- `fit(raw: Dict[str, np.ndarray], edge_index: torch.Tensor, verbose: bool = True) -> None`  
- `predict(raw: Dict[str, np.ndarray], edge_index: torch.Tensor) -> Dict[str, np.ndarray]`  
- Internal helpers: `_stack_windows(...)`, `_score_windows(...)`, `_batch_graph(ei, B, S)` (replicates `edge_index` across batch by offsetting node IDs).

---

## 9) Evaluation protocol

- **Station‑level** (window granularity): confusion matrix over windows.  
- **Network‑level** (stroke granularity): stroke counted **TP** if ≥ `min_stn` stations are hot in any overlapping window; FP are **clusters** with no overlap to truth.

**Metrics**: Precision, Recall, F1 at **network** level; per‑station P/R/F1 for diagnostics.

---

## 10) Expected behaviour

- Improves recall over per‑station CDAE when bursts are **spatially correlated**.  
- Envelope gate stabilises precision by suppressing low‑energy errors.  
- Majority smoothing reduces flicker without sacrificing much latency.

---

## 11) Risks, limitations & failure modes

- **Graph mis‑specification**: fully‑connected works but may overweight noisy neighbours; consider geography‑aware sparsification.  
- **Non‑determinism**: fix seeds and disable nondeterministic kernels if strict reproducibility is required.  
- **Compute**: Extra cost from GAT layers; still trainable in minutes for ~S≈10–20 on a laptop‑class GPU.  
- **Gate coupling**: very burst‑heavy scenes can inflate the envelope threshold (fewer windows flagged).

---

## 12) Security & privacy (secure environment)

- **Data locality**: Training/inference are in‑process; no external services.  
- **PII** (*Personally Identifiable Information*): ADC traces contain no PII; redact any side‑channel metadata upstream.  
- **Classification**: **Official Sensitive** — handle, store, and share in line with your organisation’s Official Sensitive procedures.  
- **Model artefacts**: Store weights and thresholds per model within secure perimeter only.

---

## 13) Reproducibility & versioning

- **Config hash**: `{win, hop, latent, epochs, batch, noise_std, z_mad, env_pct, graph_spec}` + code version.  
- **Artefacts to log**: seeds, training loss curves, per‑station error histograms, chosen gates, and final masks.  
- **CI**: With fixed seeds and frozen data, assert identical masks (allow for small FP differences if CuDNN is nondeterministic).

---

## 14) Dependencies & environment

- Python ≥ 3.9  
- NumPy, SciPy (Hilbert envelope)  
- PyTorch ≥ 1.13, torch-geometric ≥ 2.3 (`GATv2Conv`)  
- (Optional) CUDA or Metal (MPS)

**Resource footprint (indicative)**  
- Memory: O(batch × S × W) activations + GAT attention tensors.  
- Throughput: 3 epochs, batch 256, S≈10–12 → typically a few minutes on a laptop GPU.

---

## 15) Governance

- **Owners**: Lightning Sim · Detection  
- **On‑call**: john.goodacre@example.co.uk  
- **Escalation**: #sim-detection (internal)

---

## 16) Changelog

- **v0.1.0** — Initial Graph‑CDAE model + card (GATv2 conditioning, robust‑z + envelope gate, evaluator integration).
