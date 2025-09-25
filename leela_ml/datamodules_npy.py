"""
─────────────────────────────────────────────────────────────────────────────
 datamodules_npy.py — windowed *.npy* datasets for PyTorch / Lightning
 Author: jsg
─────────────────────────────────────────────────────────────────────────────

FILE CONTENTS
─────────────
1. **StrikeDataset** – chunks a long 1-D waveform (saved as *.npy*) into
   fixed-length windows and assigns a *binary* label to each window:
       1 → overlaps at least one simulated “lightning burst”
       0 → pure background noise.

2. **NoiseDataset** – thin wrapper around *StrikeDataset* that returns
   **only** noise windows (label==0).  Useful for denoiser AEs or GANs
   that need “clean” exemplars.

=================================================================
THE DATA FORMAT EXPECTED
=================================================================
For each recording station the simulator outputs:

    • <station>_wave.npy    # float32 1-D waveform, length S samples
    • <station>_meta.json   # {"fs": 40000, "events": [t₀, t₁, …]}

`events` are burst start-times in seconds; each burst is assumed to last
exactly **40 ms** (configurable below).

=================================================================
STRIKEDATASET  —  DETAILED SPEC
=================================================================
Init parameters
───────────────
npy_path   : str | Path
meta_path  : str | Path
chunk_size : int = 16_384      # samples per window  (≈ 0.41 s at 40 kHz)
overlap    : float = 0.5       # 0 → no overlap, 0.5 → 50 % overlap
burst_ms   : int = 40          # burst duration (ms); set to 0 for “centre-only”

Derived attributes
──────────────────
fs        : sampling rate, pulled from meta.json
hop       : int(hop) = int(chunk_size * (1-overlap))
n_win     : number of windows = floor((S - chunk_size) / hop) + 1
wave      : np.memmap  view of the raw signal  (no RAM copies)
_windows  : np.ndarray view shape (n_win, chunk_size) built with
            `numpy.lib.stride_tricks.as_strided` → **zero-copy**

Label logic (pseudocode)
────────────────────────
```
labels = np.zeros(n_win, dtype=np.float32)
for t in meta["events"]:
    s0 = int(t * fs)                       # burst start sample
    s1 = s0 + int(burst_ms * fs / 1000)    # burst end
    first = max(0,  (s0 - chunk_size) // hop + 1)
    last  = min(n_win-1,  s1 // hop)
    labels[first : last+1] = 1
```
Thus a window is **positive** if *any* part overlaps a burst.

PyTorch Dataset API
───────────────────
__len__()        → n_win
__getitem__(idx) → (x, y)

    x : torch.float32 tensor (1, chunk_size)
    y : torch.float32 scalar  (0.0 or 1.0)

RAM / IO facts
──────────────
• The *.npy* stays on disk as a memory-map – constant RAM use.
• Window generation via as_strided is O(1) and copy-free.
• First access to a window triggers a page-fault read (~4 KiB).

=================================================================
NOISEDATASET
=================================================================
Subclass of StrikeDataset that keeps an index list
`noise_idx = np.where(labels == 0)` at init.

__getitem__(i) returns **only** the signal tensor, discarding label.

Ideal for:
    * standard denoisers (learn x → x_clean)
    * negative batches in contrastive objectives

=================================================================
STRENGTHS
=================================================================
✓ Memory-efficient – handles hours-long recordings on a laptop.
✓ Overlap parameter doubles as a simple data-augmentation knob.
✓ Works out-of-the-box with `torch.utils.data.DataLoader` and
  PyTorch-Lightning datamodules.

=================================================================
WEAKNESSES & IMPROVEMENTS
=================================================================
✗ Burst duration hard-coded (40 ms) – expose via meta.json or arg.
✗ Only binary labels – extend to multi-class for severity levels.
✗ as_strided windows are non-contiguous – some exotic ops need `.copy()`.
✗ No on-the-fly normalisation – add optional `mean/std` scaling.

=================================================================
EXAMPLE USAGE
=================================================================
```python
from torch.utils.data import DataLoader
from datamodules_npy import StrikeDataset, NoiseDataset

# full strike/noise mix
ds = StrikeDataset("stationA_wave.npy", "stationA_meta.json",
                   chunk_size=16384, overlap=0.5)
dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)

# iterate
for x, y in dl:                # x:(32,1,16384)  y:(32,)
    ...

# noise-only stream
noise_ds = NoiseDataset("stationA_wave.npy", "stationA_meta.json")
noise_dl = DataLoader(noise_ds, batch_size=64, shuffle=True)
```

─────────────────────────────────────────────────────────────────────────────
"""


from pathlib import Path
import json, numpy as np
try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # allow torch-less usage
    torch = None
    class Dataset:
        pass


# ──────────────────────────────────────────────────────────────────────────────
class StrikeDataset(Dataset):
    """
    Slice a single .npy waveform into fixed-length windows and give each window
    a binary label (1 = overlaps a synthetic lightning burst, 0 = pure noise).

    Parameters
    ----------
    npy_path : str
        Path to the .npy file for one station **or** an alias *_wave.npy.
    meta_path : str
        Path to the *_meta.json file produced by the simulator.
    chunk_size : int
        Number of samples per window.
    overlap : float in [0,1)
        0   → non-overlapping windows (hop == chunk_size)
        0.5 → 50 % overlap (hop == chunk_size / 2) etc.
    """
    def __init__(self,
                 npy_path: str,
                 meta_path: str,
                 chunk_size: int = 16_384,
                 overlap:    float = 0.0):

        self.npy_path   = Path(npy_path)
        self.meta_path  = Path(meta_path)
        self.chunk_size = int(chunk_size)
        self.overlap    = float(overlap)

        # ---------- load waveform lazily (memory-mapped) ---------------------
        self.wave  = np.load(self.npy_path, mmap_mode="r")
        self.meta  = json.load(open(self.meta_path))
        self.fs    = self.meta["fs"]                           # sample rate
        self.hop   = int(self.chunk_size * (1 - self.overlap))
        if self.hop <= 0:
            raise ValueError("overlap must be < 1.0")

        self.n_win = 1 + (len(self.wave) - self.chunk_size) // self.hop
        self._windows = np.lib.stride_tricks.as_strided(
            self.wave,
            shape   =(self.n_win, self.chunk_size),
            strides =(self.wave.strides[0]*self.hop, self.wave.strides[0]),
        ).astype("f4", copy=False)

        # ---------- build labels (1 if ANY overlap with a burst) -------------
        labels = np.zeros(self.n_win, dtype=np.uint8)
        for ev in self.meta["events"]:
            s0 = int(ev["t"] * self.fs)                    # start sample
            s1 = s0 + int(0.04 * self.fs)                 # +40 ms burst
            first = max(0,       (s0 - self.chunk_size) // self.hop + 1)
            last  = min(self.n_win - 1,  s1 // self.hop)
            labels[first:last+1] = 1
        self.labels = labels

    # ---------- PyTorch Dataset API ------------------------------------------
    def __len__(self) -> int:
        return self.n_win

    def __getitem__(self, idx: int):
        x = self._windows[idx]          # numpy view, shape (chunk_size,)
        if torch is None:
            return x.reshape(1, -1).astype(np.float32), np.float32(self.labels[idx])
        # add channel-dim → (1,T)  ;  convert to torch.float32
        return (
            torch.from_numpy(x).unsqueeze(0),
            torch.tensor(float(self.labels[idx])),
        )


# ── helper view that exposes only noise windows (label == 0) ------------------
class NoiseDataset(StrikeDataset):
    """Use this for noise-only training."""
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.noise_idx = np.where(self.labels == 0)[0]

    def __len__(self): return len(self.noise_idx)

    def __getitem__(self, i):
        # i is local index into noise-only array
        return super().__getitem__(int(self.noise_idx[i]))[0]   # drop label
