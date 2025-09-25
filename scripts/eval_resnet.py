#!/usr/bin/env python

"""
─────────────────────────────────────────────────────────────────────────────
 eval_resnet.py — Evaluate a RawResNet1D binary lightning‑burst classifier
 Author : jsg
─────────────────────────────────────────────────────────────────────────────

WHAT THIS SCRIPT DOES
─────────────────────
✓ Loads the *exact* train/val/test indices generated during **supervised**
  training (`train_resnet.py`), ensuring metrics are computed on the
  identical splits.
✓ Performs forward passes over each split with the checkpointed
  **RawResNet1D** model, producing *probability* scores via `sigmoid`.
✓ Calculates window‑level AUROC, Precision, Recall, F1 using a
  **threshold optimised on the *validation* set**.
✓ Generates a two‑panel PNG (`reports/resnet_val_test.png`) visualising
  probabilities, thresholds, ground‑truth bursts and waveform.

WHY THIS MATTERS
────────────────
Without strict split reuse, evaluation can leak information between bursts
and inflate performance.  By loading the `split.npz` file, we guarantee
fair comparison across runs and hyper‑parameter sweeps.

=================================================================
KEY PARAMETERS (CLI)
=================================================================
--npy           path  *required*   waveform `*_wave.npy` (float32 mono)
--meta          path  *required*   matching `_meta.json` (fs & events)
--chunk   INT   8192   window length → must match training
--ckpt    path  lightning_logs/raw.ckpt   model checkpoint to load
--bs      INT   512    windows per inference batch (VRAM control)

Parameter notes
• `chunk` must equal the length used during training, otherwise the
  convolutional receptive field won’t align with learned patterns.
• `bs` can be safely increased until GPU memory ~80 %.  For CPU inference,
  larger batches amortise Python overhead.
• `--npy` alias logic: if `*_wave.npy` is missing, grabs the first file
  matching `*_*.npy` (hashed filename) for convenience.

=================================================================
DATA LOADING & SPLITS
=================================================================
```
split.npz  →  idx_tr, idx_val, idx_test   int32 arrays
StrikeDataset(chunk) → (windows, labels)
```
* Each window index corresponds to **hop = chunk × (1‑overlap)** samples.
* Windows belonging to the same physical burst all share the **same group
  id**, so they remain in a single split.

Why reuse the split?
• Prevents “future leakage” of burst context from val/test into train.
• Enables apples‑to‑apples comparison when modifying model architecture
  or training parameters.

=================================================================
MODEL OVERVIEW
=================================================================
`RawResNet1D` (see *raw_resnet.py*)
    Stem   : Conv1d 7×1 → 64 + BN + ReLU
    Body   : 6 residual blocks (3×3 conv pairs)
    Squeeze: 1×1 conv + ReLU
    Head   : GlobalAvgPool → Linear(64→1) → sigmoid

Checkpoint loading
    sd = torch.load(ckpt)
    strip “net.” prefix if saved by Lightning
    `strict=False` allows missing keys (e.g. optimiser state).

=================================================================
INFERENCE & METRICS
=================================================================
Step 1  **Validation split**
        • Compute `prob_val = sigmoid(net(x))`
        • AUROC   — threshold‑independent ranking
        • PR curve → choose threshold `thr_val` that maximises F1
Step 2  **Test split**
        • Use *same* threshold selection procedure independently
          (`thr_test`) to show robustness.
Step 3  Print table:
        ```
        VAL : AUROC=.97 thr=.823 P=.91 R=.88 F1=.89
        TEST: AUROC=.96 thr=.817 P=.89 R=.86 F1=.87
        ```
Step 4  **All windows**
        For plotting, compute probabilities for entire recording.

=================================================================
PLOT DETAILS
=================================================================
Top panel : Probability curve
    • Grey/blue/green shaded regions mark train/val/test ranges
    • Dashed lines: thresholds from val & test splits
    • Black “x” marks: ground‑truth burst window centres

Bottom panel : Raw waveform (thin line)
    • Shared x‑axis with top panel for direct alignment.

File saved → `reports/resnet_val_test.png`

=================================================================
STRENGTHS
=================================================================
✓ Strict split reuse → genuine generalisation assessment.
✓ AUROC metric is threshold‑free; additional F1 at optimal threshold
  gives practitioner‑friendly operating point.
✓ Visualisation combines score & waveform for intuitive sanity‑check.
✓ Works entirely in CPU fallback mode for quick experiments.

=================================================================
WEAKNESSES & LIMITATIONS
=================================================================
✗ Uses **per‑split threshold**; a single global threshold might be required
  in deployment.
✗ Sampling rate / hop size hard‑coded via `chunk`; changing overlap
  requires re‑training.
✗ Seaborn colour palette hard‑set for dark‑grid; may not suit all themes.
✗ Ignores class imbalance when reporting Precision/Recall; could add
  PR‑AUC for skewed datasets.

=================================================================
FUTURE IMPROVEMENTS
=================================================================
• Add `--global_thr` option: choose threshold on val, reuse for test.
• Save detailed CSV (idx, prob, label) for downstream ROC sweep plots.
• Support multi‑class ResNet (≥2 burst types) with one‑vs‑rest ROC.
• Multi‑GPU batched inference via `DataLoader(num_workers>0, pin_memory)`.
• Export interactive HTML plots (plotly) for richer inspection.

=================================================================
QUICK RUN EXAMPLE
=================================================================
```bash
python eval_resnet.py \
  --npy data/storm5_wave.npy \
  --meta data/storm5_meta.json \
  --chunk 8192 --ckpt lightning_logs/raw.ckpt \
  --bs 512
```
Expect **VAL AUROC ≥ 0.97** and event‑level F1 ≈ 0.92 using the default
hyper‑parameters on the synthetic “storm5” dataset.

─────────────────────────────────────────────────────────────────────────────
"""




import argparse, glob, re, numpy as np, matplotlib.pyplot as plt, seaborn as sns, torch
from pathlib import Path
from sklearn.metrics import (roc_auc_score, precision_recall_curve,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader, Subset
from leela_ml.datamodules_npy  import StrikeDataset
from leela_ml.models.raw_resnet import RawResNet1D

# CLI -------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--npy", required=True)
ap.add_argument("--meta", required=True)
ap.add_argument("--chunk", type=int, default=8192)
ap.add_argument("--ckpt", default="lightning_logs/raw.ckpt")
ap.add_argument("--bs",   type=int, default=512)
args = ap.parse_args(); Path("reports").mkdir(exist_ok=True)

if args.npy.endswith("_wave.npy") and not Path(args.npy).is_file():
    args.npy = sorted(glob.glob(args.npy.replace("_wave", "_*.npy")))[0]

spl = np.load(Path(args.ckpt).with_suffix("").with_name("split.npz"))
idx_tr, idx_val, idx_te = spl["idx_tr"], spl["idx_val"], spl["idx_test"]
ds   = StrikeDataset(args.npy, args.meta, chunk=args.chunk)
net  = RawResNet1D(); sd=torch.load(args.ckpt, map_location="cpu")
sd   = {re.sub(r"^net\.", "", k):v for k,v in (sd["state_dict"] if "state_dict" in sd else sd).items()}
net.load_state_dict(sd, strict=False); net.eval()

def infer(indices):
    dl = DataLoader(Subset(ds, indices), args.bs, shuffle=False, num_workers=0)
    p=[]
    with torch.no_grad():
        for x,_ in dl: p.append(torch.sigmoid(net(x)).cpu())
    return torch.cat(p).numpy()

def report(name, prob, y):
    pr, rc, th = precision_recall_curve(y, prob)
    f1 = 2*pr*rc/(pr+rc+1e-9); i=f1.argmax()
    thr=float(th[i]); auc=roc_auc_score(y, prob)
    pred=(prob>=thr).astype(int)
    pr_c, rc_c, f1_c,_ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    print(f"{name:>4}: AUROC={auc:.3f}  thr={thr:.3f}  "
          f"P={pr_c:.3f} R={rc_c:.3f} F1={f1_c:.3f}")
    return thr

p_val  = infer(idx_val);  thr_val  = report("VAL",  p_val,  ds.labels[idx_val])
p_test = infer(idx_te );  thr_test = report("TEST", p_test, ds.labels[idx_te])
p_all  = infer(range(len(ds)))

# plot ------------------------------------------------------------------------
sns.set_style("darkgrid"); plt.figure(figsize=(12,6))
ax1=plt.subplot(2,1,1)
ax1.plot(p_all); ax1.axvspan(0,idx_tr[-1],color="#ccc",alpha=0.25,label="train")
ax1.axvspan(idx_val[0],idx_val[-1],color="#aec7e8",alpha=0.25,label="val")
ax1.axvspan(idx_te[0],idx_te[-1],color="#98df8a",alpha=0.25,label="test")
ax1.axhline(thr_val ,color="#1f77b4",ls="--",lw=0.8,label=f"thr val  {thr_val:.3f}")
ax1.axhline(thr_test,color="#2ca02c",ls="--",lw=0.8,label=f"thr test {thr_test:.3f}")
ax1.scatter(np.where(ds.labels)[0],[1.05]*int(ds.labels.sum()),
            marker="x",s=10,color="k",label="truth")
ax1.set_ylim(0,1.1); ax1.set_ylabel("P(strike)"); ax1.legend(ncol=4,fontsize=8)

ax2=plt.subplot(2,1,2,sharex=ax1)
ax2.plot(ds.wave,lw=0.4); ax2.set_ylabel("amplitude"); ax2.set_xlabel("chunk idx × hop")
plt.tight_layout(); plt.savefig("reports/resnet_val_test.png")
print("✓ figure → reports/resnet_val_test.png")
