---
title: "Engineering Playbook — Python Good Practices in the Lightning‑Storm Project"
status: "guide"
audience: "Engineers & Researchers (Python / ML)"
version: "1.0"
---

> **TL;DR**  
> This guide distils **Python engineering practices** demonstrated across the modularised lightning‑storm project: clean data contracts, feature registry design, safe windowing, reproducibility, consistent `fit`/`predict` interfaces, evaluation decoupling, and pragmatic ML explainability. Each section shows **idiomatic snippets** from the codebase and a **naïve counter‑example** explaining why the chosen approach is better.

---

## Contents

1. [Project structure & separation of concerns](#project-structure--separation-of-concerns)  
2. [Data contracts: `StormConfig` & `StormBundle`](#data-contracts-stormconfig--stormbundle)  
3. [Feature engineering: registry pattern](#feature-engineering-registry-pattern)  
4. [Windowing: memory & performance](#windowing-memory--performance)  
5. [Model interfaces: consistent `fit`/`predict`](#model-interfaces-consistent-fitpredict)  
6. [Evaluation: strict, reusable, decoupled](#evaluation-strict-reusable-decoupled)  
7. [Reproducibility & randomness](#reproducibility--randomness)  
8. [Robustness to API drift](#robustness-to-api-drift)  
9. [Explainability: feature importance without labels](#explainability-feature-importance-without-labels)  
10. [Production hardening opportunities](#production-hardening-opportunities)  
11. [Appendix: Idioms & patterns](#appendix-idioms--patterns)
12. [Python & NumPy Snippets](#Python--NumPy-Snippets)

---

## Project structure & separation of concerns

**Why it’s good:** Each module has a single, clear responsibility and a **clean interface** to the next stage.

```
lightning_sim/
  sim/generator.py          # StormConfig, StormGenerator → StormBundle
  features/basic.py         # make_windows, feature registry, iso13/ae10/iso16
  evaluation/window_eval.py # EvalConfig, evaluate_windowed_model()
  detectors/                # Hilbert, NCD, IsoForest, ExtendedIsoForest,
                           # CDAE, GraphCDAE, OCSVM — all share fit/predict
```

- **Simulation** produces a typed container (`StormBundle`) — no model logic here.
- **Features** expose atomic and composite features via a **registry**, not `if/elif` ladders.
- **Detectors** implement **the same public methods** and return the **same output shape**.
- **Evaluation** consumes *only* the common mask interface, so you can swap models freely.

> This pipeline makes it trivial to benchmark detectors *apples to apples* by holding the evaluator constant.

---

## Data contracts: `StormConfig` & `StormBundle`

Use **dataclasses** (with `frozen=True` for config) to express **intent, typing and immutability**.

```python
from dataclasses import dataclass
from typing import Dict, List
import numpy as np, pandas as pd

@dataclass(frozen=True)
class StormConfig:
    seed: int = 424242
    duration_min: int = 5
    scenario: str = "medium"     # "near" | "medium" | "far"
    difficulty: int = 1          # 1 … 9
    snr_db: float = 1.0
    fs: int = 109_375
    bits: int = 14
    vref: float = 1.0

@dataclass
class StormBundle:
    quantised: Dict[str, np.ndarray]
    events: List[dict]
    stroke_records: List[dict]
    df_wave: pd.DataFrame
    df_labels: pd.DataFrame
```

**Why this is better than ad‑hoc dicts**

- **Type hints** guide IDEs and linters; catching mistakes early.
- **Frozen config** prevents accidental mutation of key parameters mid‑run.
- The **return container** keeps related artefacts together; avoids “loose globals”.

**Naïve alternative (don’t do this):**

```python
# Anti-pattern: magic strings & mutable config
CFG = {"seed": 424242, "fs": 109375}
def generate(cfg):
    return {"q": q, "e": events, "sr": strokes}  # terse, opaque keys
```

Problems: lack of types, mutable global state, brittle keys, hard to test.

---

## Feature engineering: registry pattern

A **decorator‑based registry** makes adding features safe and discoverable, without editing core code.

```python
_feature_funcs = {}

def register_feature(name: str):
    def _wrap(func):
        _feature_funcs[name] = func
        return func
    return _wrap

@register_feature("hilbert_peak")
def feat_hilbert_peak(win_mat, ctx):
    env = np.abs(hilbert(win_mat.astype(float), axis=1))
    return env.max(axis=1, keepdims=True)

def list_features():
    return sorted(_feature_funcs)
```

**Benefits**

- **Open‑closed principle**: new features via `@register_feature`, core stays untouched.
- **Runtime discoverability**: `list_features()` backs CLI/UX choices and tests.
- **Uniform signature**: every feature accepts `(win_mat, ctx)` → shape `(n, d)`.

**Naïve alternative:**

```python
def compute_features(name, win_mat, ctx):
    if name == "hilbert_peak": ...
    elif name == "rms_db": ...
    # grows forever, merge conflicts, hard to test per-feature
```

---

## Windowing: memory & performance

Two flavours appear in the project:

1) **Index‑based windows** (safe, copies):

```python
def make_windows(sig: np.ndarray, win: int, hop: int) -> np.ndarray:
    n = (len(sig) - win) // hop + 1
    idx = np.arange(0, n*hop, hop)[:, None] + np.arange(win)
    return sig[idx]  # copies; safe for mutation and persistence
```

2) **Stride‑view windows** (fast, zero‑copy — handle with care):

```python
def _win_view(sig: np.ndarray, W: int, H: int) -> np.ndarray:
    n = (len(sig) - W) // H + 1
    return np.lib.stride_tricks.as_strided(
        sig, shape=(n, W), strides=(sig.strides[0]*H, sig.strides[0])
    )
```

**Guidance**

- Use **index‑based** windows in feature extractors where safety > peak speed.
- Use **stride‑views** in tight loops (e.g., NCD compression) to avoid extra memory — *but never write to them*, and ensure the base array is **C‑contiguous**.

**Naïve alternative:** building Python lists in a loop — slow and memory hungry.

```python
# Anti-pattern: repeated slicing in Python loops
wins = []
for i in range(0, len(sig)-win+1, hop):
    wins.append(sig[i:i+win])
win_mat = np.stack(wins)  # slow & GC overhead
```

---

## Model interfaces: consistent `fit`/`predict`

All detectors follow a **shared contract** to simplify evaluation and orchestration.

```python
class IsoForestModel:
    def __init__(self, *, win=1024, hop=512, contamination=0.0015, n_trees=150, random_state=42):
        self.win, self.hop = win, hop
        self.fx = FeatureExtractor(["iso16"])
        self.models = {}  # station → (scaler, model, aux)

    def fit(self, raw: Dict[str, np.ndarray], fs: int, verbose=True) -> None:
        feats, _ = self.fx.transform(raw, win=self.win, hop=self.hop, fs=fs)
        # … fit scaler + model per station; store in self.models

    def predict(self, raw: Dict[str, np.ndarray], fs: int) -> Dict[str, np.ndarray]:
        feats, _ = self.fx.transform(raw, win=self.win, hop=self.hop, fs=fs)
        # … return {station: bool mask of shape (n_win,)}
```

**What makes this good**

- **Same I/O type** across models (`Dict[str, bool array]`) → drop‑in replaceable.
- Features are **owned by the model** (via `self.fx`) → no hidden globals.
- Hyper‑parameters live in `__init__`; `fit` has **no surprising side effects**.

**Naïve alternative:** mixing training and evaluation in the same method, changing global config, or returning mismatched shapes.

---

## Evaluation: strict, reusable, decoupled

`evaluate_windowed_model` converts stroke‑level truth into **window masks** per station, aligns predictions, and computes **station** and **network** metrics.

Key practices:
- **Typed config** (`EvalConfig`) with sensible defaults (burst length auto‑derived from `fs`).  
- **Tolerance dilation** (`tol_win`) for latency‑robustness experiments.  
- **Clustered FP** counting at network level — prevents over‑penalising long buzz periods.

```python
@dataclass
class EvalConfig:
    win:int=1024; hop:int=512; fs:int=109_375
    burst_len:int|None=None; min_stn:int=2; tol_win:int=0
    def __post_init__(self):
        if self.burst_len is None:
            self.burst_len = int(0.04 * self.fs)
```

**Naïve alternative:** recomputing evaluation rules inside each model (drift, bugs, incomparable metrics).

---

## Reproducibility & randomness

- **Single RNG**: `np.random.default_rng(cfg.seed)` per generator/model run.  
- **Printed run summary**: difficulty tier, SNR, scenario, counts — makes audit trails easy.  
- ML components (IsoForest, CDAE) seed their frameworks where possible.

```python
self.rng = np.random.default_rng(cfg.seed)
# torch.manual_seed(...), model random_state=42, etc.
```

**Anti‑pattern:** using global `np.random` without a fixed seed → non‑reproducible artefacts.

---

## Robustness to API drift

The extended Isolation‑Forest handles version changes in `isotree` with a **guarded accessor**:

```python
def _avg_depth(iso, X):
    for kw in ({"type": "avg_depth"}, {"output_type": "avg_depth"}):
        try:
            return iso.predict(X, **kw)
        except TypeError:
            continue
    return -iso.predict(X)  # fallback
```

**Why it’s good:** keeps notebooks stable across library upgrades; encapsulates the variance in one helper.

**Naïve alternative:** assuming one API forever → brittle notebooks.

---

## Explainability: feature importance without labels

When native importances are absent, the project falls back to **permutation importance** over the **average depth** score — label‑free and aligned to the detector’s internal notion of “normality”.

```python
def _perm_importance(iso, Xs, depth_base, n_iter=5, subsample=5000):
    imp = np.zeros(Xs.shape[1])
    for _ in range(n_iter):
        for j in range(Xs.shape[1]):
            X_shuf = Xs.copy()
            np.random.shuffle(X_shuf[:, j])
            depth_shuf = _avg_depth(iso, X_shuf)
            imp[j] += np.mean(np.abs(depth_shuf - depth_base))
    s = imp.sum()
    return imp / s if s else imp
```

**Why it’s good:** simple, robust, **normalised** across stations; clarifies which features (e.g., envelope peak vs spectral bands) drive anomaly decisions.

---

## Production hardening opportunities

1. **Logging, not printing**  
   Replace `print()` with `logging` and **structured run metadata**.
   ```python
   import logging; log = logging.getLogger(__name__)
   log.info("Flashes=%d strokes=%d samples=%d", len(events), n_strk, N)
   ```

2. **Windowing unification**  
   Provide a single, unit‑tested `window_view(sig, win, hop, *, mode="copy|view")` with shape/stride assertions.

3. **Validation**  
   Use `pydantic` (or `dataclasses` + custom validators) to enforce `difficulty in 1..9`, `fs > 0`, station codes present, etc.

4. **Config serialisation**  
   Persist `StormConfig`, detector hyper‑parameters, thresholds and seeds alongside artefacts (`.json`), with a **config hash** for traceability.

5. **Parallelism**  
   Per‑station model fits can parallelise (joblib / multiprocessing) since inputs are independent.

6. **Score calibration**  
   Move from fixed percentiles to **stable quantile estimators** or fit a simple **Gaussian Mixture** on reconstruction errors for better thresholds under burst‑heavy scenes.

7. **Device guards**  
   In Torch models, add deterministic flags and fallbacks for non‑deterministic kernels if exact reproducibility is required.

---

## Appendix: Idioms & patterns

### A. Clean “flags” computation from a single integer

```python
def _difficulty_flags(d):
    return dict(
        ic_mix=d>=2, multipath=d>=3, coloured_noise=d>=4, rfi_tones=d>=5,
        impulsive_rfi=d>=6, sprite_ring=d>=5, false_transient=d>=6,
        clipping=d>=5, multi_cell=d>=6, skywave=d>=7, sferic_bed=d>=7,
        clock_skew=d>=8, gain_drift=d>=8, dropouts=d>=8, low_snr=d>=9, burst_div=d>=9
    )
```

**Why it’s good:** single source of truth → coherent scenarios.

---

### B. Chunked processing for large arrays

```python
chunk = int(20*FS)
for s0 in range(0, N, chunk):
    e0 = min(N, s0+chunk)
    seg = base_noise(s0, e0)
    # inject bursts falling into [s0:e0)
    # filter & quantise per chunk
```

**Why it’s good:** bounded memory, stream‑friendly, easier to parallelise.

---

### C. Percentile thresholds (station‑specific) vs fixed cut‑offs

```python
thr = np.percentile(score_vec, 99.9)   # adapts to station gain/noise
mask = score_vec > thr
```

**Why it’s good:** normalises across amplitude differences and drift.

---

### D. Majority smoothing to reduce flicker

```python
from scipy.signal import convolve
mask = convolve(mask.astype(int), [1,1,1], mode="same") >= 2
```

**Why it’s good:** removes isolated flips; preserves onset timing.

---

### E. Torch inference with batching & no grad

```python
with torch.no_grad():
    for i0 in range(0, len(win_mat), 4096):
        seg = torch.from_numpy(win_mat[i0:i0+4096].astype(np.float32)/32768.0)[:, None].to(device)
        rec = net(seg).cpu().numpy()
        errs[i0:i0+len(rec)] = ((rec - seg.cpu().numpy())**2).mean(axis=(1,2))
```

**Why it’s good:** fast, memory‑aware, portable across CPU/GPU.

---

### F. Clean evaluator error handling

```python
n_win = min((len(q[s])-win)//hop + 1 for s in station_order)
if n_win <= 0:
    raise RuntimeError("No complete windows to score")
```

**Why it’s good:** fails early with a clear message.

---

## Python & NumPy Snippets

> This section explains a handful of **non‑trivial patterns** used in the project, shows a **naïve alternative** for contrast, and suggests **advanced optimisation** paths (NumPy/Numba/CuPy).

---

### 1) Vectorised window indexing (no Python loops)

**Project pattern**
```python
def make_windows(sig: np.ndarray, win: int, hop: int) -> np.ndarray:
    n = (len(sig) - win) // hop + 1
    idx = np.arange(0, n*hop, hop)[:, None] + np.arange(win)
    return sig[idx]  # shape: (n, win)
```

**Naïve alternative**
```python
wins = []
for i in range(0, len(sig)-win+1, hop):
    wins.append(sig[i:i+win])
win_mat = np.stack(wins, axis=0)
```

**Why the vectorised version is better**
- Single NumPy expression ⇒ fewer Python interpreter trips, better cache locality.
- `idx` leverages broadcasting to produce all start/offset indices at once.

**Advanced**
- For very large signals, consider `numpy.lib.stride_tricks.sliding_window_view(sig, win)[::hop]` (safer view) or `as_strided` (see §2) with **read‑only** discipline.
- Preallocate downstream arrays with `np.empty((n, win), dtype=...)` to avoid temporary lists.

---

### 2) Zero‑copy window views (stride trick) — with guard rails

**Project pattern (NCD fast path)**
```python
def _win_view(sig, W, H):
    n = (len(sig) - W) // H + 1
    return np.lib.stride_tricks.as_strided(
        sig, shape=(n, W), strides=(sig.strides[0]*H, sig.strides[0])
    )
```

**Naïve alternative**
```python
# Copying every window (slow, memory heavy)
win_mat = np.array([sig[i:i+W] for i in range(0, len(sig)-W+1, H)])
```

**Why the view is better**
- **Zero copy**; builds a logical 2‑D view on the same memory.
- Essential when compression/FFT runs per window.

**Safety checklist**
- Only **read** from the view; never write (aliasing).
- Ensure `sig.flags['C_CONTIGUOUS']` is `True`.
- Optionally wrap in an assertion helper:

```python
def safe_win_view(sig, W, H):
    assert sig.ndim == 1 and sig.flags.c_contiguous
    n = (len(sig) - W) // H + 1
    view = np.lib.stride_tricks.as_strided(
        sig, shape=(n, W), strides=(sig.strides[0]*H, sig.strides[0])
    )
    view.setflags(write=False)  # guard writes
    return view
```

**Advanced**
- When available, prefer `sliding_window_view` (NumPy ≥1.20) — same benefits with fewer foot‑guns.

---

### 3) Bit‑level encoding with `np.diff` + `np.packbits`

**Project pattern (NCD “bits” encoder)**
```python
def _enc_bits(arr: np.ndarray) -> bytes:
    diff = np.diff(arr.astype(np.int16), prepend=arr[0])
    return np.packbits((diff > 0).astype(np.uint8)).tobytes()
```

**Naïve alternative**
```python
bits = bytearray()
last = int(arr[0])
byte = 0; k = 0
for x in map(int, arr[1:]):
    b = 1 if x - last > 0 else 0
    byte = (byte << 1) | b; k += 1
    if k == 8: bits.append(byte); byte = 0; k = 0
    last = x
if k: bits.append(byte << (8-k))
payload = bytes(bits)
```

**Why the vectorised version is better**
- Pure NumPy ufuncs (`diff`, `>`, `packbits`) push work into C; orders‑of‑magnitude faster.
- No branchy Python loop ⇒ consistent throughput.

**Advanced**
- With **CuPy**, `diff`/`packbits` are available on GPU; test with `import cupy as np` drop‑in for large batches.

---

### 4) Spectral features in one shot (batch RFFT)

**Current approach (per‑window loop, OK for clarity)**
```python
out = np.empty((win_mat.shape[0], 1))
for i, w in enumerate(win_mat):
    f, P = welch(w, fs, nperseg=256)
    P /= P.sum() + 1e-9
    out[i, 0] = -np.sum(P * np.log(P + 1e-9))
```

**Vectorised alternative (faster for large batches)**
```python
X = np.fft.rfft(win_mat.astype(float), axis=1)     # (n, Nf)
P = (X.real**2 + X.imag**2)                        # power
P /= P.sum(axis=1, keepdims=True) + 1e-9
spec_entropy = -(P * np.log(P + 1e-9)).sum(axis=1, keepdims=True)
```

**Why it helps**
- One batched FFT replaces thousands of small ones.
- Enables cheap **centroid/bandwidth** via `np.einsum`:

```python
freqs = np.fft.rfftfreq(win_mat.shape[1], d=1/fs)  # (Nf,)
Pn = P / (P.sum(axis=1, keepdims=True) + 1e-12)
centroid  = np.einsum('nf,f->n', Pn, freqs)
bandwidth = np.sqrt(np.einsum('nf,f->n', Pn, (freqs-centroid[:,None])**2))
```

**Advanced**
- Use **pocketfft** backend (NumPy ≥1.20) automatically.
- On GPU, use **CuPy**’s `cupy.fft.rfft` with the same code.

---

### 5) Haversine: scalar clarity vs vector speed

**Project pattern (readable scalar version)**
```python
def _hav(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, φ2 = map(math.radians, (lat1, lat2))
    dφ, dλ = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2*R*math.asin(math.sqrt(a))
```

**Vectorised alternative (when broadcasting arrays)**
```python
def haversine_vec(lat1, lon1, lat2, lon2):
    R = 6371.0
    la1, la2 = np.radians(lat1), np.radians(lat2)
    dφ, dλ   = np.radians(lat2-lat1), np.radians(lon2-lon1)
    a = np.sin(dφ/2)**2 + np.cos(la1)*np.cos(la2)*np.sin(dλ/2)**2
    return 2*R*np.arcsin(np.sqrt(a))
```

**Trade‑off**
- Keep the **scalar** version inside tight per‑stroke loops (small constants).
- Switch to **vector** when computing distances for many stations/flashes at once.

---

### 6) Preallocation beats appending

**Project pattern**
```python
ncd = np.empty(n_win, np.float32)
for i in range(n_win):
    ncd[i] = ...
```

**Naïve alternative**
```python
vals = []
for i in range(n_win):
    vals.append(compute(...))
ncd = np.array(vals, dtype=np.float32)
```

**Why preallocate**
- Fewer allocations and copies; predictable memory footprint.

**Advanced**
- For pure‑Python loops with numeric kernels, consider **Numba** JIT:
```python
from numba import njit

@njit(cache=True, fastmath=True)
def crest_factor(seg):
    s2 = 0.0
    peak = 0.0
    for x in seg:
        y = float(x)
        s2 += y*y
        if y*y > peak: peak = y*y
    return math.sqrt(peak) / (math.sqrt(s2/len(seg)) + 1e-12)
```

---

### 7) Robust numerics: tiny epsilons

**Project pattern**
```python
rms = np.sqrt((win_mat.astype(float)**2).mean(axis=1)) + 1e-9
logp = P * np.log(P + 1e-9)         # avoid log(0)
ratio = peak / (median + 1e-9)      # avoid /0
```

**Naïve alternative**
```python
rms = np.sqrt((win_mat.astype(float)**2).mean(axis=1))
logp = P * np.log(P)    # can be -inf
```

**Why add eps**
- Prevents `NaN`/`±inf` that contaminate downstream metrics and plots.
- Choose eps in context (e.g., `1e-12` for power, `1e-9` for envelope).

---

### 8) Caching expensive pure functions

**Project pattern (NCD compression sizes)**
```python
from functools import lru_cache

@lru_cache(maxsize=None)
def _c_size(payload: bytes) -> int:
    return len(bz2.compress(payload, 9))
```

**Naïve alternative**
```python
def _c_size(payload: bytes) -> int:
    return len(bz2.compress(payload, 9))  # recompress duplicates each time
```

**Why cache**
- Byte‑identical windows arise often (“bits” encoding) ⇒ instant cache hits.
- Memory is modest (payload hashed by `lru_cache` key).

**Advanced**
- If memory is tight, bound the cache (`maxsize=100_000`) or use a **content hash** (e.g., `xxhash`) as key.

---

### 9) Chunked streaming keeps memory bounded

**Project pattern**
```python
chunk = int(20*FS)
for s0 in range(0, N, chunk):
    e0 = min(N, s0+chunk)
    seg = base_noise(s0, e0)
    # inject bursts, filter, quantise in-place
```

**Naïve alternative**
```python
# Build full signal in RAM and process at once
seg = build_everything(N)  # huge temporary arrays
```

**Why stream**
- Avoids peak memory spikes; aligns with file‑writer/dataset chunking.

**Advanced**
- Use `numpy.memmap` to spill very large arrays to disk without loading all at once.

---

### 10) End‑to‑end GPU/CuPy sketch (drop‑in acceleration)

**Concept**
```python
USE_GPU = False
try:
    import cupy as xp
    USE_GPU = True
except Exception:
    import numpy as xp  # fallback to NumPy

# Same code works: xp.fft.rfft, xp.abs, xp.log, xp.einsum
X = xp.fft.rfft(win_mat.astype(xp.float32), axis=1)
P = (X.real**2 + X.imag**2)
P /= P.sum(axis=1, keepdims=True) + 1e-9
spec_entropy = -(P * xp.log(P + 1e-9)).sum(axis=1)
```
- **Caveat**: SciPy routines (e.g., `welch`, `hilbert`) are CPU‑only; use CuPy equivalents or keep those on CPU.

---

### 11) Parallelism patterns (beyond “production hardening”)

- **Per‑station** independence ⇒ parallel map with `joblib.Parallel` or `multiprocessing.Pool`:
```python
from joblib import Parallel, delayed
res = Parallel(n_jobs=-1)(delayed(process_station)(nm, sig) for nm, sig in raw.items())
```
- **Vector‑first**: before parallelising, ensure heavy math is vectorised/batched (FFT example in §4). This often brings larger wins than threads.

---

### 12) Safer index maths for window truth

**Project pattern (careful rounding & clipping)**
```python
s0, s1 = rec["sample_idx"], rec["sample_idx"] + burst_len - 1
w0 = max(0, int(np.ceil((s0 + 1 - win) / hop)))  # inclusive
w1 = min(n_win-1, int(np.floor(s1 / hop)))       # inclusive
truth[stn][w0:w1+1] = True
```

**Naïve alternative**
```python
w0 = s0 // hop
w1 = (s0 + burst_len) // hop
truth[stn][w0:w1] = True  # off-by-ones & boundary misses
```

**Why the careful version**
- Aligns window *boundaries* precisely; avoids systematic early/late labelling.

---

## Advanced optimisation roadmap (CPU, Numba, GPU)

- **Batch FFT everywhere**: migrate per‑window frequency moments (centroid, bandwidth, entropy) to batched RFFT/Einsum (§4). Expect 5–20× speed‑ups for large batches.
- **Numba JIT for pure‑Python loops**: crest factors, STA/LTA, and compression prep can be `@njit`‑compiled (avoid SciPy calls inside JIT).
- **Use `sliding_window_view`** where supported; retain `as_strided` only when you’ve profiled and added write‑guards.
- **CuPy drop‑in**: allow optional GPU acceleration for FFT‑heavy steps; keep a **clean backend shim** (`xp = numpy | cupy`).
- **Memory mapping** for multi‑minute storms: `np.memmap` per station to bound RSS and enable chunked processing.
- **Vectorise encoders**: where possible, move byte‑encoders to vectorised NumPy and cache (`lru_cache`) payload sizes; profile compressors (`bz2`, `lz4`, `zstd`) for better CPU throughput.
- **Quantile/percentile**: for very long runs, consider **TDigest** or **P² quantile estimator** to compute thresholds online without holding all scores.

> **Rule of thumb:** *Vectorise first, batch transforms second, only then add threads/GPUs.*


## Final thought

The project attempts to balance **clarity** (typed containers, registries, consistent interfaces) with **some performance** (chunking, batch scoring) and **reproducibility** (seeded RNGs, stateless evaluators). The same patterns generalise to other time‑series problems: keep the **contracts crisp**, the **data flow explicit**, and the **evaluation decoupled**.

