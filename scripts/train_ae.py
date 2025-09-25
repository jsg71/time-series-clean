#!/usr/bin/env python
"""
─────────────────────────────────────────────────────────────────────────────
 train_ae.py — Train a 1‑D U‑Net Denoising Auto‑Encoder on *noise‑only* windows
 Author : jsg
─────────────────────────────────────────────────────────────────────────────

WHY THIS SCRIPT?
────────────────
Legacy baseline ⇢ learn an **Auto‑Encoder (AE)** that *only* sees background
noise.  When we later run it over the full recording, any lightning burst
(or other anomaly) should produce a large reconstruction error because it
doesn’t fit the learned “noise manifold”.

High‑level flow
───────────────
1. Load StrikeDataset from a *.npy* waveform + *.json* meta.            fileciteturn6file0
2. Slice into windows ➟ time‑based 60 / 20 / 20 split.
3. Keep **noise windows** (label==0) for train + val.
4. Train `UNet1D` AE (depth,base from CLI) with L1 loss.
5. Save best checkpoint (lowest *val_mae*) and the split indices
   so evaluation scripts can reuse the exact partition.

=================================================================
COMMAND‑LINE ARGUMENTS
=================================================================
--npy      PATH   **required**  waveform *.npy* (alias *_wave.npy allowed)
--meta     PATH   **required**  matching meta.json
--chunk    INT    4096     window length (samples)
--overlap  FLOAT  0.5      0 ≤ overlap < 1  (0.5 ⇒ hop = 50 %)
--bs       INT    128      batch size
--epochs   INT    20       max epochs (no early‑stop by default)
--depth    INT    4        U‑Net depth (encoder/decoder levels)
--base     INT    16       channels in first encoder block
--lr       FLOAT  1e‑3     Adam learning rate
--device   auto|cpu|mps|cuda   auto → CUDA ▸ MPS ▸ CPU
--ckpt     PATH   lightning_logs/ae_best.ckpt  output checkpoint file

Argument rationale
──────────────────
chunk   : 4096 samples @ 40 kHz ≈ 102 ms — big enough to hold burst + context
overlap : 0.5 doubles dataset size w/out big memory cost
depth   : 4 ⇒ receptive field ±(9×2⁴)=±144 ≈ 3.6 ms per conv path
lr      : 1e‑3 stable for Adam on MAE; adjust with `--lr` when changing `base`
device  : “auto” keeps script portable across CPU‑only laptops and GPUs

=================================================================
DATASET & SPLITS
=================================================================
StrikeDataset             (see datamodules_npy.py)
    windows : (n, chunk) view via numpy.as_strided  (zero‑copy)
    labels  : 0 / 1 — produced from burst timestamps

Sequential split 60‑20‑20 **by index (time order)**
⋯ prevents future windows leaking into past during validation.

Noise subset
    idx_tr_noise = intersect(train_full, where(label==0))
    idx_val_noise = intersect(val_full, where(label==0))

Why sequential?
Burst density can change across time (e.g. storm build‑up); random split
would let the model “peek” at future distributions.

=================================================================
MODEL
=================================================================
UNet1D                (leela_ml.models.dae_unet: depth, base set via CLI)
LightningModule       (LitAE) wraps the network + loss + metrics

Training‑step  : L1Loss(recon,x)  (aka MAE)
Validation‑step: logs val_l1 + updates MeanAbsoluteError metric
Optimiser      : Adam(lr) — no scheduler for simplicity

Checkpointing  : save **best** model w.r.t. `val_mae`
Split indices  : *.split.npz* stores arrays {idx_train, idx_val, idx_test}

=================================================================
EXPECTED RESULTS (synthetic data, base=16 depth=4)
=================================================================
| Dataset     | val_mae ≈ | epochs |
|-------------|-----------|--------|
| stationA    | 0.012     | 18‑20  |
| stationB    | 0.015     | 20     |

Look for reconstruction‑error spikes ≥ 6×MAD at burst times.

=================================================================
STRENGTHS
=================================================================
✓ Single command reproducibility.
✓ Time‑based split mimics real deployment chronology.
✓ L1 loss encourages sharp reconstructions (vs MSE blur).
✓ No GPU dependency; runs on CPU albeit slower.

=================================================================
WEAKNESSES & IMPROVEMENTS
=================================================================
✗ Training discards burst examples entirely → model may “hallucinate” them
  as noise at inference.  *Fix*: denoise objective with paired data.
✗ Adam with fixed LR; consider CosineAnnealing, OneCycleLR.
✗ No **early stopping** → wasted epochs if plateau hit early.
✗ BatchNorm inside UNet1D brittle on tiny `--bs`; swap to GroupNorm.
✗ Single‑run log dir; parallel sweeps overwrite unless `--ckpt` changed.

Future ideas
────────────
• STFT‑based multi‑loss (time‑domain L1 + freq‑domain L1).
• t‑distributed latent regulariser (Variational AE).
• Mixed‑precision (`torch.set_float32_matmul_precision("high")`) for speed.
• Online augmentation: Gaussian noise scale jitter, phase shuffle.

=================================================================
EXAMPLE COMMAND
=================================================================
```bash
python train_ae.py \
  --npy data/stationA_wave.npy \
  --meta data/stationA_meta.json \
  --chunk 4096 --overlap 0.5 \
  --bs 128 --epochs 30 --depth 4 --base 16 --lr 1e-3 \
  --device cuda \
  --ckpt models/ae_stationA.ckpt
```

─────────────────────────────────────────────────────────────────────────────
"""

import argparse, glob, numpy as np, torch, pytorch_lightning as pl, shutil
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MeanAbsoluteError
from torch.nn import L1Loss
from leela_ml.datamodules_npy import StrikeDataset
from leela_ml.models.dae_unet import UNet1D

# --- Parse command-line arguments ---
p = argparse.ArgumentParser(description="Train 1D U-Net autoencoder on noise windows")
p.add_argument("--npy",     required=True, help="Path to waveform .npy file (or alias *_wave.npy)")
p.add_argument("--meta",    required=True, help="Path to metadata .json file for the waveform")
p.add_argument("--chunk",   type=int,   default=4096, help="Window length (samples)")
p.add_argument("--overlap", type=float, default=0.5,  help="Window overlap fraction (0 ≤ overlap < 1)")
p.add_argument("--bs",      type=int,   default=128,  help="Batch size")
p.add_argument("--epochs",  type=int,   default=20,   help="Number of training epochs")
p.add_argument("--depth",   type=int,   default=4,    help="U-Net depth (number of down/up levels)")
p.add_argument("--base",    type=int,   default=16,   help="U-Net base number of filters")
p.add_argument("--lr",      type=float, default=1e-3, help="Learning rate for Adam optimizer")
p.add_argument("--device",  choices=["auto","cpu","mps","cuda"], default="auto", help="Compute device")
p.add_argument("--ckpt",    default="lightning_logs/ae_best.ckpt", help="Output path for best model checkpoint")
args = p.parse_args()
Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)

# --- Handle alias path (if provided *_wave.npy alias) ---
if args.npy.endswith("_wave.npy") and not Path(args.npy).exists():
    # If alias not found, try to find actual file (e.g. first station file)
    cand = sorted(glob.glob(args.npy.replace("_wave", "_*.npy")))
    if not cand:
        raise FileNotFoundError(f"Waveform file {args.npy} not found.")
    args.npy = cand[0]  # use the first matching station file

# --- Load dataset and prepare windows ---
ds      = StrikeDataset(args.npy, args.meta, chunk_size=args.chunk, overlap=args.overlap)
windows = ds._windows              # shape (n_windows, chunk_size)
labels  = ds.labels               # binary labels per window (0=noise, 1=burst)
n_win   = len(ds)
print(f"[info] total windows = {n_win:,} (chunk={args.chunk}, overlap={args.overlap})")

# Train/Val/Test split indices (time-based sequential split 60/20/20)
i1 = int(0.6 * n_win); i2 = int(0.8 * n_win)
idx_tr_full  = np.arange(0, i1)
idx_val_full = np.arange(i1, i2)
idx_te_full  = np.arange(i2, n_win)
print(f"[info] full windows train/val/test = {len(idx_tr_full)}/{len(idx_val_full)}/{len(idx_te_full)}")

# Use only noise windows (label=0) for training and validation
noise_idx    = np.where(labels == 0)[0]
idx_tr_noise = np.intersect1d(idx_tr_full, noise_idx)
idx_val_noise= np.intersect1d(idx_val_full, noise_idx)
print(f"[info] noise windows used for train/val = {len(idx_tr_noise)}/{len(idx_val_noise)}")

# Prepare PyTorch datasets and loaders
X_tr = torch.from_numpy(windows[idx_tr_noise]).unsqueeze(1)  # shape (N,1,chunk)
X_va = torch.from_numpy(windows[idx_val_noise]).unsqueeze(1)
tr_ds = TensorDataset(X_tr)
va_ds = TensorDataset(X_va)
dl_tr = DataLoader(tr_ds, batch_size=args.bs, shuffle=True,  num_workers=0)
dl_va = DataLoader(va_ds, batch_size=args.bs, shuffle=False, num_workers=0)

# --- LightningModule definition for training ---
class LitAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net  = UNet1D(depth=args.depth, base=args.base)
        self.loss = L1Loss()               # L1 reconstruction loss
        self.mae  = MeanAbsoluteError()    # MAE metric for validation
    def forward(self, x):
        return self.net(x)
    def training_step(self, batch, batch_idx):
        (x,) = batch
        rec = self(x)
        loss = self.loss(rec, x)
        self.log("train_l1", loss, prog_bar=False)
        return loss
    def validation_step(self, batch, batch_idx):
        (x,) = batch
        rec = self(x)
        # Log validation loss and update MAE
        self.log("val_l1", self.loss(rec, x), prog_bar=False)
        self.mae.update(rec, x)
    def on_validation_epoch_end(self):
        # Log average MAE on validation set (for checkpointing)
        val_mae = self.mae.compute()
        self.log("val_mae", val_mae, prog_bar=True)
        self.mae.reset()
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=args.lr)

# --- Configure training on specified device (CPU, GPU, or MPS) ---
accel = args.device
if accel == "auto":
    if torch.cuda.is_available():
        accel = "gpu"
    elif torch.backends.mps.is_available():
        accel = "mps"
    else:
        accel = "cpu"
# Use single device (no distributed training for simplicity)
trainer = pl.Trainer(
    max_epochs=args.epochs,
    accelerator=accel,
    devices=1,
    logger=False,
    callbacks=[pl.callbacks.ModelCheckpoint(dirpath=Path(args.ckpt).parent,
                                           filename=Path(args.ckpt).stem,
                                           monitor="val_mae", mode="min", save_top_k=1)],
    enable_progress_bar=True,
    enable_model_summary=False,
    num_sanity_val_steps=0
)
print(f"[info] Training on {accel.upper()} for {args.epochs} epochs...")
trainer.fit(LitAE(), dl_tr, dl_va)

# ── save the best model checkpoint ───────────────────────────────
best_ckpt = Path(trainer.checkpoint_callback.best_model_path)
dst_ckpt  = Path(args.ckpt)

if best_ckpt.is_file():
    # copy only if source ≠ destination
    if best_ckpt.resolve() != dst_ckpt.resolve():
        shutil.copy(best_ckpt, dst_ckpt)
        print(f"✓ Best AE model copied → {dst_ckpt}")
    else:
        print(f"✓ Best AE model already at {dst_ckpt}")
else:  # training produced no “best” (very rare)
    torch.save(LitAE().state_dict(), dst_ckpt)
    print(f"✓ No best-model path; saved current weights → {dst_ckpt}")


# Save indices of splits for later evaluation (to compute AUROC on val/test)
split_path = dst_ckpt.with_suffix("").with_name(dst_ckpt.stem + ".split.npz")
np.savez(split_path, idx_train=idx_tr_full.astype(int), idx_val=idx_val_full.astype(int), idx_test=idx_te_full.astype(int))
print(f"✓ Saved data split indices → {split_path}")
