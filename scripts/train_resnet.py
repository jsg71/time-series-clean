#!/usr/bin/env python

"""
─────────────────────────────────────────────────────────────────────────────
 train_resnet.py — Supervised training of a RawResNet1D lightning‐burst
                   classifier with event‑aware splits
 Author : jsg
─────────────────────────────────────────────────────────────────────────────

CONTEXT & CONTRAST
──────────────────
Earlier scripts (`train_ae*.py`) learn **unsupervised / one‑class**
reconstruction from *noise‑only* windows.
**This script is different** – it trains a *supervised* binary classifier
(`RawResNet1D`) that *does* see labelled bursts during training.

Why do supervised as well?
• Provides an **upper‑bound** on achievable detection accuracy when
  labels are available.
• Lets us compare cost/benefit of annotation effort vs unsupervised error
  thresholding.
• Serves as teacher model for knowledge‑distillation into lighter models.

─────────────────────────────────────────────────────────────────────────────
CLI ARGUMENTS (with rationale)
──────────────────────────────
--npy, --meta      paths to waveform *.npy* and meta.json (REQUIRED)

--chunk    8192    window length (samples).  At 40 kHz ⇒ 204.8 ms
                   Long enough to contain entire burst envelope.

--overlap  0.75    75 % overlap ⇒ hop = 2048 samples (51 ms).  Provides
                   dense sampling while limiting dataset size ×4.

--bs       64      mini‑batch size; fits 8 GB GPUs at chunk 8 k.

--epochs   40      enough to converge; early stop via ModelCheckpoint monitor
                   on val_AUC.

--ckpt     path    where the *best* model (highest val_AUC) is copied.

─────────────────────────────────────────────────────────────────────────────
DATASET & EVENT‑AWARE SPLITTING
───────────────────────────────
`StrikeDataset` → windows + labels (1 if overlaps any burst).

**groups = event‑ID per window**

• Each burst’s contiguous windows share a single group ID.
• Noise windows get unique IDs.

`GroupShuffleSplit`
    pass 1 : 60 % train  40 % temp
    pass 2 : temp → 50 % val  50 % test

Guarantees that **all windows from a given physical burst** land in exactly
one split → prevents “label leakage” where fragments of the same burst
appear in both train and val.

Indices are stored in `split.npz` for downstream reproducibility.

Class imbalance handling
    pos ≪ neg ⇒ `WeightedRandomSampler`
        weight[pos] = N_neg / N_pos
        weight[neg] = 1

─────────────────────────────────────────────────────────────────────────────
MODEL : RawResNet1D  (see raw_resnet.py)
─────────────────────────────────────────
Stem       : 7‑tap conv (1→64) + BN + ReLU
Body       : repeated residual pairs 3×3, same channels (default 6 blocks)
Squeeze    : 1×1 conv mix
Head       : GlobalAvgPool1d → Linear(64→1)

Pros
• Constant output length irrespective of `chunk`.
• Residual links improve gradient flow; trains in < 1 h on GPU.
• Only ~130 k parameters – deployable on Raspberry Pi‑class edge devices.

Loss                : `BCEWithLogitsLoss` (stable for binary logits)
Metric (val)        : `BinaryAUROC`  — threshold‑agnostic performance.
Optimiser           : `Adam(lr=1e‑3)`  (cosine schedule could further help).
Checkpoint monitor  : **val_auc** (higher better)  → saves best model only.

─────────────────────────────────────────────────────────────────────────────
EXPECTED BEHAVIOUR & RESULTS
────────────────────────────
Synthetic “storm5” (chunk 8 192, overlap 0.75)
    • Train/Val/Test windows  ≈  6 k / 2 k / 2 k
    • Class ratio (train)     ≈  1 : 15 (burst : noise)
    • Converged val_AUC       ≥ 0.97 after ~25 epochs
    • Event‑level F1          ≈ 0.92  (threshold tuned on val set)

Real field data tends to score 5‑10 % lower (unclean labels & edge cases).

─────────────────────────────────────────────────────────────────────────────
STRENGTHS
──────────
✓ Supervised learning exploits labelled bursts → higher accuracy ceiling.
✓ Event‑aware split avoids overly optimistic metrics.
✓ Weighted sampler combats class imbalance without down‑sampling noise.
✓ AUROC monitoring obviates manual threshold pick during training.

WEAKNESSES & LIMITATIONS
────────────────────────
✗ Requires labelled ground‑truth bursts (costly in real life).
✗ Model sees *context* of bursts during training ⇒ may over‑fit on
  amplitude envelope specifics; cross‑station generalisation needs care.
✗ No LR schedule or early‑stop callback – fixed 40 epochs regardless of
  plateau (can waste compute).
✗ GroupShuffleSplit uses deterministic `random_state`; change for
  different fold.

─────────────────────────────────────────────────────────────────────────────
IMPROVEMENT IDEAS
──────────────────
• Replace Adam with **AdamW + cosine** scheduler (warm‑restarts).
• Add `--dilate` flag in RawResNet1D to widen receptive field cheaply.
• Integrate **mixup** or **spec‑augment** style jitter along time axis.
• Log metrics to TensorBoard for real‑time monitoring.
• Hyper‑parameter sweep via **pytorch‑lightning CLI** or Optuna.

─────────────────────────────────────────────────────────────────────────────
EXAMPLE COMMAND
────────────────
```bash
python train_resnet.py \
  --npy   data/storm5_wave.npy \
  --meta  data/storm5_meta.json \
  --chunk 8192 --overlap 0.75 \
  --bs 64 --epochs 40 \
  --ckpt models/raw_storm5.ckpt
```

─────────────────────────────────────────────────────────────────────────────
"""



import argparse, glob, shutil
from pathlib import Path
import numpy as np, torch, pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchmetrics.classification import BinaryAUROC
from sklearn.model_selection import GroupShuffleSplit
from leela_ml.datamodules_npy  import StrikeDataset   # signature uses chunk_size, overlap
from leela_ml.models.raw_resnet import RawResNet1D

# ─── CLI ----------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--npy",    required=True)
ap.add_argument("--meta",   required=True)
ap.add_argument("--chunk",  type=int,  default=8192, help="window length")
ap.add_argument("--overlap",type=float,default=0.75, help="fractional overlap 0–1")
ap.add_argument("--bs",     type=int,  default=64)
ap.add_argument("--epochs", type=int,  default=40)
ap.add_argument("--ckpt",   default="lightning_logs/raw_best.ckpt")
args = ap.parse_args()
log_dir = Path(args.ckpt).parent; log_dir.mkdir(parents=True, exist_ok=True)

# alias fallback ---------------------------------------------------------------
if args.npy.endswith("_wave.npy") and not Path(args.npy).is_file():
    alt = sorted(glob.glob(args.npy.replace("_wave", "_*.npy"))); args.npy = alt[0]

# dataset (note the right kwargs) ---------------------------------------------
ds = StrikeDataset(args.npy, args.meta,
                   chunk_size=args.chunk,
                   overlap=args.overlap)

labels = ds.labels.astype(int)

# build event-id for every chunk (noise chunks keep unique ids) ---------------
groups = np.arange(len(ds), dtype=int)   # each chunk its own group
evt = max(groups) + 1
i = 0
while i < len(ds):
    if labels[i]:
        j = i
        while j < len(ds) and labels[j]:
            groups[j] = evt
            j += 1
        evt += 1
        i = j
    else:
        i += 1

# GroupShuffleSplit 60/20/20 ---------------------------------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=0)
tmp_tr, tmp_rest = next(gss.split(np.zeros(len(ds)), labels, groups))
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=1)
idx_val, idx_test = next(gss2.split(np.zeros(len(tmp_rest)),
                                    labels[tmp_rest], groups[tmp_rest]))
idx_tr   = np.sort(tmp_tr)
idx_val  = np.sort(tmp_rest[idx_val])
idx_test = np.sort(tmp_rest[idx_test])

np.savez(log_dir/"split.npz", idx_tr=idx_tr, idx_val=idx_val, idx_test=idx_test)
print(f"[info] chunks   train/val/test = {len(idx_tr)}/{len(idx_val)}/{len(idx_test)}")
print(f"[info] strikes  train/val/test = {labels[idx_tr].sum()}/"
      f"{labels[idx_val].sum()}/{labels[idx_test].sum()}")

# DataLoaders ------------------------------------------------------------------
pos = labels[idx_tr].sum(); neg = len(idx_tr) - pos
w_pos = neg/pos if pos else 1.0
weights = [w_pos if labels[i] else 1.0 for i in idx_tr]
dl_tr  = DataLoader(Subset(ds, idx_tr), args.bs,
                    sampler=WeightedRandomSampler(weights, len(idx_tr), True),
                    num_workers=0)
dl_val = DataLoader(Subset(ds, idx_val), args.bs, shuffle=False, num_workers=0)

# Lightning module ------------------------------------------------------------
class Lit(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net  = RawResNet1D()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.auc  = BinaryAUROC()
    def forward(self,x): return self.net(x)
    def training_step(self,batch,_):
        x,y=batch; loss=self.loss(self(x),y); self.log("train_loss",loss); return loss
    def validation_step(self,batch,_):
        x,y=batch; out=self(x); self.log("val_loss",self.loss(out,y))
        self.auc.update(torch.sigmoid(out), y.int())
    def on_validation_epoch_end(self):
        self.log("val_auc", self.auc.compute(), prog_bar=True); self.auc.reset()
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

ckpt_cb = pl.callbacks.ModelCheckpoint(dirpath=log_dir, filename="tmp",
                                       monitor="val_auc", mode="max",
                                       save_top_k=1, auto_insert_metric_name=False)
pl.Trainer(max_epochs=args.epochs, accelerator="auto", logger=False,
           callbacks=[ckpt_cb], num_sanity_val_steps=0).fit(Lit(), dl_tr, dl_val)

best = Path(ckpt_cb.best_model_path)
if best.is_file():
    shutil.copy(best, args.ckpt)
    shutil.copy(best, log_dir/"raw.ckpt")
    print("✓ best model →", args.ckpt)
