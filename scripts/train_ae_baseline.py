#!/usr/bin/env python

"""
─────────────────────────────────────────────────────────────────────────────
 train_ae_baseline.py — Train a baseline Denoising Auto‑Encoder (U‑Net1D)
 Author : jsg
─────────────────────────────────────────────────────────────────────────────

GOAL
────
• Learn to reconstruct **pure‑noise windows** of a long field recording.
• Later, anomaly detectors can flag bursts by measuring reconstruction error
  (high error ⇒ window differs from the noise manifold).

WHY THIS SCRIPT EXISTS
──────────────────────
Legacy experiments assumed *no model sees any burst data during training*.
That guarantees a reconstruction‑error spike whenever an unseen lightning
burst corrupts a window.  The script reproduces that set‑up exactly so new
models can be benchmarked against the same baseline.

=================================================================
CLI ARGUMENTS
=================================================================
--npy      <path>   REQUIRED  path to waveform *.npy*  (or *\*_wave.npy alias)
--meta     <path>   REQUIRED  path to matching meta.json produced by simulator
--chunk    <int>   window length in samples                 (default 4096)
--overlap  <float> window overlap fraction 0–<1             (default 0.5)
--bs       <int>   mini‑batch size                          (default 64)
--epochs   <int>   training epochs                          (default 20)
--ckpt     <path>  where to save best Lightning checkpoint   default:
                   lightning_logs/ae_best.ckpt

Tip: the alias rule means you can pass
    --npy stationA_wave.npy
even if the simulator actually wrote `stationA_<hash>.npy`; the script picks
the first glob match.

=================================================================
DATA PIPELINE
=================================================================
Step 1  Load **all** windows (noise + burst) via `StrikeDataset`
        windows  →  windows[n_win, chunk]    (numpy view, zero‑copy)
        labels   →  0 (noise) / 1 (burst)

Step 2  Sequential 60/20/20 split on the *index order* (time order):
        • Train_full   idx  0   …  i₁‑1
        • Val_full     idx  i₁ …  i₂‑1
        • Test_full    idx  i₂ …  end
        This avoids “future leaking into past” across the split.

Step 3  For **train** and **val** we keep *only noise windows*
        idx_tr_noise = intersect(train_full, where(label==0))
        idx_val_noise = intersect(val_full,   where(label==0))

Resulting tensor shapes
    X_tr : (n_tr_noise,   1, chunk)
    X_va : (n_val_noise,  1, chunk)

=================================================================
MODEL — `UNet1D` auto‑encoder
=================================================================
• Architecture defined in `leela_ml.models.dae_unet_baseline`, depth=4.
• Maps (1, chunk) → (1, chunk); no bottleneck loss.

Lightning wrapper (`LitAE`)
───────────────────────────
training_step
    L1 = mean |recon − x|  (alias for MAE)
validation_step
    logs val_L1 and updates `torchmetrics.MeanAbsoluteError`
configure_optimisers
    Adam(lr = 1e‑3)

=================================================================
CHECKPOINTING
=================================================================
`pl.callbacks.ModelCheckpoint`
    • Monitored metric : **val_mae** (lower = better)
    • Saves to `dirpath = Path(ckpt).parent`, filename stem same as target
After training the best ckpt is **copied** to `--ckpt` path, ensuring a
stable filename downstream.

Additionally, the script writes a `*.split.npz` alongside the ckpt
containing the indices (int32) for **train/val/test full windows** so that
evaluation scripts can load exactly the same partition.

=================================================================
REASONING BEHIND DESIGN
=================================================================
✓ Training only on noise windows enforces *one‑class* reconstruction.
✓ Sequential split mimics deployment (model sees future it never trained on).
✓ L1 / MAE is robust and directly comparable across signals.
✓ Simpler than MSE : less sensitive to outliers and bumps.

=================================================================
STRENGTHS
=================================================================
• End‑to‑end reproducible with one command.
• Memory‑efficient — windows are numpy *views*; no extra copies.
• Automatic checkpoint & split file facilitates later evaluation.
• No dependency on GPUs — runs fine on CPU for small datasets.

=================================================================
WEAKNESSES & IMPROVEMENTS
=================================================================
✗ Noise‑only training discards useful “context” from burst windows; a
  *denoising* objective (x_noisy → x_clean) could learn richer features.
✗ L1 loss equally penalises all frequencies; perceptual losses (e.g. STFT
  magnitude) might correlate better with audible distortion.
✗ Fixed Adam LR 1e‑3 — could benefit from cosine LR schedule or LR finder.
✗ No early stopping — training continues for `--epochs` even if val error
  plateaus; add `EarlyStopping(monitor='val_mae', patience=5)`.
✗ Single‑run log dir; multiple runs overwrite older checkpoints unless you
  change `--ckpt`.

=================================================================
EXAMPLE COMMAND
=================================================================
```bash
python train_ae_baseline.py \
  --npy   data/stationA_wave.npy \
  --meta  data/stationA_meta.json \
  --chunk 4096 --overlap 0.5 \
  --bs 64 --epochs 30 \
  --ckpt models/ae_stationA.ckpt
```

=================================================================
ARCHITECTURAL IDEAS FOR FUTURE WORK
=================================================================
• Replace `UNet1D` with a **TFC‑TDF** block (dilated convs) to enlarge
  receptive field without extra pooling.
• Freeze encoder after noise training; fine‑tune decoder on limited burst
  examples → semi‑supervised anomaly detection.
• Integrate **mixup** on noise windows to further diversify the input
  manifold.
• Swap to **bfloat16** training on modern GPUs for speed & memory gain.
─────────────────────────────────────────────────────────────────────────────
"""



import argparse, glob, numpy as np, torch, pytorch_lightning as pl, shutil
from pathlib import Path
from torch.utils.data           import DataLoader, TensorDataset
from torchmetrics               import MeanAbsoluteError
from torch.nn                   import L1Loss
from leela_ml.datamodules_npy   import StrikeDataset
from leela_ml.models.dae_unet_baseline   import UNet1D

p = argparse.ArgumentParser()
p.add_argument("--npy",     required=True)
p.add_argument("--meta",    required=True)
p.add_argument("--chunk",   type=int,   default=4096)
p.add_argument("--overlap", type=float, default=0.5)
p.add_argument("--bs",      type=int,   default=64)
p.add_argument("--epochs",  type=int,   default=20)
p.add_argument("--ckpt",    default="lightning_logs/ae_best.ckpt")
args = p.parse_args()
Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)

# alias fallback
if args.npy.endswith("_wave.npy") and not Path(args.npy).exists():
    cand = sorted(glob.glob(args.npy.replace("_wave","_*.npy")))
    if not cand: raise FileNotFoundError()
    args.npy = cand[0]

# load full dataset (all windows, all labels)
ds       = StrikeDataset(args.npy, args.meta,
                         chunk_size=args.chunk,
                         overlap=args.overlap)
windows  = ds._windows    # shape (n_win, chunk)
labels   = ds.labels      # 0=noise, 1=burst
n_win    = len(ds)

# 60/20/20 sequential split on full windows
i1 = int(0.6 * n_win)
i2 = int(0.8 * n_win)
idx_tr_full = np.arange(0, i1)
idx_val_full= np.arange(i1, i2)
idx_te_full = np.arange(i2, n_win)
print(f"[info] full windows tr/va/te = {len(idx_tr_full)}/{len(idx_val_full)}/{len(idx_te_full)}")

# within train/val, pick only noise windows for training & validation
noise = np.where(labels==0)[0]
# intersect noise with each split
idx_tr_noise = np.intersect1d(idx_tr_full, noise)
idx_val_noise= np.intersect1d(idx_val_full, noise)
print(f"[info] noise windows tr/va = {len(idx_tr_noise)}/{len(idx_val_noise)}")

# build TensorDatasets for AE training
X_tr = torch.from_numpy(windows[idx_tr_noise]).unsqueeze(1)
X_va = torch.from_numpy(windows[idx_val_noise]).unsqueeze(1)
tr_ds = TensorDataset(X_tr)
va_ds = TensorDataset(X_va)

dl_tr = DataLoader(tr_ds, batch_size=args.bs, shuffle=True,  num_workers=0)
dl_va = DataLoader(va_ds, batch_size=args.bs, shuffle=False, num_workers=0)

# LightningModule -------------------------------------------------------------
class LitAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net  = UNet1D()
        self.loss = L1Loss()
        self.mae  = MeanAbsoluteError()
    def forward(self, x): return self.net(x)
    def training_step(self, batch, _):
        (x,) = batch
        rec = self(x); l = self.loss(rec, x)
        self.log("train_l1", l); return l
    def validation_step(self, batch, _):
        (x,) = batch
        rec = self(x)
        self.log("val_l1", self.loss(rec, x))
        self.mae.update(rec, x)
    def on_validation_epoch_end(self):
        self.log("val_mae", self.mae.compute(), prog_bar=True)
        self.mae.reset()
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

ckpt_cb = pl.callbacks.ModelCheckpoint(
    dirpath=Path(args.ckpt).parent,
    filename=Path(args.ckpt).stem,
    monitor="val_mae", mode="min", save_top_k=1
)
trainer = pl.Trainer(
    max_epochs=args.epochs,
    accelerator="auto",
    logger=False,
    callbacks=[ckpt_cb],
    num_sanity_val_steps=0
)
trainer.fit(LitAE(), dl_tr, dl_va)

# copy best ckpt safely
best = Path(ckpt_cb.best_model_path).resolve()
dst  = Path(args.ckpt).resolve()
if best.exists() and best != dst:
    shutil.copy(best, dst)
    print(f"✓ AE saved → {dst}")
else:
    print(f"✓ AE checkpoint at → {dst}")

# save full tr/va/te splits for eval
split_path = dst.with_suffix("").with_name(dst.stem + ".split.npz")
np.savez(split_path,
         idx_tr=idx_tr_full.astype(int),
         idx_val=idx_val_full.astype(int),
         idx_test=idx_te_full.astype(int))
print(f"✓ splits saved → {split_path}")
