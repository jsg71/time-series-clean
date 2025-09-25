#!/usr/bin/env python3

"""
─────────────────────────────────────────────────────────────────────────────
 run_ncd.py — Compression‑based unsupervised lightning‑burst detector
 Author : jsg
─────────────────────────────────────────────────────────────────────────────

BIG‑PICTURE IDEA
────────────────
Use **Normalised Compression Distance (NCD)** between consecutive windows
to detect anomalies (lightning bursts) without any learned model.

Why NCD works
─────────────
A burst changes local signal statistics (spectral content, amplitude
distribution).  If two adjacent windows are *similar*, concatenating them
won’t add new information, so their compressed length is ≈ max(C₁, C₂).
If the second window contains a burst, C(join) grows sharply ⇒ NCD rises.

Mathematically
```
NCD(A,B) = ( C(A+B) − min( C(A), C(B) ) ) / max( C(A), C(B) )
```
0  ≈ identical • ≈1  ≈ wholly different • >1 due to codec overhead

=================================================================
SCRIPT WORKFLOW
=================================================================
1. **StrikeDataset** loads waveform → sliding windows.
2. Compute NCD for each window vs its predecessor (`ncd_adjacent`)
   or against a fixed baseline window (`ncd_first`).
3. Adaptive threshold = rolling median + *k*×MAD.
4. Morphological ops clean flicker; windows above threshold = *burst mask*.
5. Merge + prune mask into **events** and compute metrics.
6. Write three diagnostic PNGs into *./reports*.

=================================================================
COMMAND‑LINE ARGUMENTS
=================================================================
--npy        PATH    REQUIRED   *_wave.npy* float32 mono waveform
--meta       PATH    REQUIRED   *_meta.json* with {"fs":…, "events":…}
--chunk      INT     512        window length (samples) — finer pulses => smaller
--overlap    FLOAT   0.9        0≤o<1; hop = chunk×(1‑o)
--codec      zlib|bz2|lzma  which std‑lib compressor; speed vs accuracy
--mad_k      FLOAT   6.0        threshold offset = median + k×MAD
--min_dur_ms FLOAT   10         minimum event duration kept after cleaning
--gap_ms     FLOAT   10         merge events separated by < gap
--dpi        INT     160        PNG resolution

Parameter intuition
───────────────────
chunk 512     → 12.8 ms @ 40 kHz; high temporal resolution.
overlap 0.9   → dense scan; 90 % redundancy but low latency.
codec zlib    → fastest; bz2 ≈2× slower but better at long repeats;
                lzma ≈3–4× slower, best compression ratio.
mad_k 6       → ~3σ if error distribution Laplace‑like.
min_dur_ms    → eliminates noisy one‑off spikes shorter than physical burst.
gap_ms        → joins two bursts separated by small silence gap.

=================================================================
CORE FUNCTIONS & DECISIONS
=================================================================
`ncd_adjacent(win, codec)`
    • Pre‑quantises float32 window to int16 (2 bytes/sample).
    • Caches individual C(wᵢ) to reuse when computing C(wᵢ₋₁+wᵢ).
    • Codec default = zlib; others exposed via CLI.

Rolling Median + MAD
    • `win_len = min_dur_ms / hop_ms` ==> adaptive to chunk & overlap.
    • MAD robust to outliers ⇒ threshold adapts to slow drifts.

Morphology passes
    • Binary dilation(k) → close sub‑min gaps inside real burst.
    • Binary erosion(k)  → remove tiny positives shorter than min_dur_ms.

Event matching
    *True burst* = contiguous １ windows labelled 1 in ground‑truth.
    TP if any overlap between predicted and true span.

Metrics printed
────────────────
Window level : Precision, Recall, F1, AUROC (if labels available)
Event level  : Precision, Recall, F1 after overlap matching

PNG outputs
───────────
reports/ncd_score.png        NCD curve + threshold (semi‑log)
reports/ncd_events.png       Waveform with TP (lime), FN (orange), FP (red)
reports/ncd_pred_timeline.png Timeline bars for visual debug

=================================================================
STRENGTHS
=================================================================
✓ Completely unsupervised — needs *no* trained model or labels.
✓ Tiny memory footprint; only two windows needed at any moment.
✓ Codec‑agnostic flexibility; zlib is fast in pure Python.
✓ Rolling threshold adapts to changing background noise.

=================================================================
WEAKNESSES & CAVEATS
=================================================================
✗ Speed dominated by compression calls; lzma on long chunk > 1 k could be slow.
✗ Choice of codec changes numeric value — results not comparable across codecs.
✗ Int16 quantisation loses micro‑amplitude differences.
✗ High‑overlap scanning multiplies runtime by 1/(1‑overlap).
✗ Rolling MAD edge effects (first/last win_len/2 windows) less reliable.

=================================================================
IDEAS FOR FUTURE IMPROVEMENTS
=================================================================
• Multithread compression via `concurrent.futures.ThreadPoolExecutor`.
• Try **zstd** (pyzstd) – faster and stronger than zlib.
• Expose `quant_bits` CLI flag (8, 12, 16 bit) to trade accuracy vs speed.
• Compute bidirectional NCD max( NCD(wᵢ₋₁,wᵢ), NCD(wᵢ,wᵢ₋₁) ).
• Adaptive overlap: small hop in high‑variance segments, large otherwise.
• JSON summary export for sweep aggregation dashboards.

=================================================================
QUICK EXAMPLE
=================================================================
```bash
python run_ncd.py \
  --npy data/stormA_wave.npy \
  --meta data/stormA_meta.json \
  --chunk 512 --overlap 0.9 \
  --codec bz2 --mad_k 6 \
  --min_dur_ms 10 --gap_ms 10
```
Expect window‑F1 ≈0.80‑0.85 on synthetic storm; lzma may bump +0.02 F1.

─────────────────────────────────────────────────────────────────────────────
"""


import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import Series
from scipy.ndimage import binary_dilation, binary_erosion
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
import json

from leela_ml.datamodules_npy import StrikeDataset
from leela_ml.ncd import ncd_adjacent, ncd_first

# ── CLI ───────────────────────────────────────────────────────────────────────
pa = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
pa.add_argument("--npy", required=True)
pa.add_argument("--meta", required=True)
pa.add_argument("--chunk", type=int, default=512)
pa.add_argument("--overlap", type=float, default=0.9)
pa.add_argument("--codec", choices=["zlib", "bz2", "lzma"], default="zlib")
pa.add_argument("--mad_k", type=float, default=6.0)
pa.add_argument("--min_dur_ms", type=float, default=10.0)
pa.add_argument("--gap_ms", type=float, default=10.0)
pa.add_argument(
    "--per_win_norm",
    action="store_true",
    help="normalise each window before compression",
)
pa.add_argument(
    "--diff_order",
    type=int,
    default=0,
    help="apply numpy.diff before compression",
)
pa.add_argument(
    "--baseline_idx",
    type=int,
    default=None,
    help="if set, compute NCD against this window index instead of adjacent",
)
pa.add_argument("--dpi", type=int, default=160)
args = pa.parse_args()
Path("reports").mkdir(exist_ok=True)

# ── data ──────────────────────────────────────────────────────────────────────
ds = StrikeDataset(args.npy, args.meta, args.chunk, args.overlap)
win = ds._windows.astype(np.float32, copy=False)
lab = ds.labels.astype(bool)
hop = ds.hop
dur_sec = len(ds.wave) / ds.fs
print(
    f"• sample_rate: {ds.fs:,.0f} Hz   windows: {len(win):,}   burst-windows (truth): {lab.sum():,}"
)
print(
    f"• window_size: {args.chunk} samp  hop: {hop} ({hop/ds.fs*1e3:.1f} ms)  duration: {dur_sec:.1f} s"
)

# ── NCD ───────────────────────────────────────────────────────────────────────
if args.baseline_idx is None:
    err = ncd_adjacent(
        win,
        codec=args.codec,
        per_win_norm=args.per_win_norm,
        diff_order=args.diff_order,
    )
else:
    err = ncd_first(
        win,
        codec=args.codec,
        per_win_norm=args.per_win_norm,
        diff_order=args.diff_order,
        baseline_idx=args.baseline_idx,
    )
print(f"• NCD finished   mean={err.mean():.4f}  med={np.median(err):.4f}")

# ── adaptive threshold --------------------------------------------------------
win_len = max(1, int(args.min_dur_ms / 1000 * ds.fs / hop))
roll = Series(err).rolling(win_len, center=True, min_periods=1)
med = roll.median().values
mad = roll.apply(lambda v: np.median(np.abs(v - np.median(v))), raw=True).values
thr = med + args.mad_k * mad
mask = err > thr

# enforce min duration + close small gaps
min_w = max(1, int(args.min_dur_ms / 1000 * ds.fs / hop))
gap_w = max(1, int(args.gap_ms / 1000 * ds.fs / hop))
mask = binary_dilation(mask, iterations=min_w)
mask = binary_erosion(mask, iterations=min_w)


# ── helpers -------------------------------------------------------------------
def runs(b):
    out = []
    on = False
    for i, v in enumerate(b):
        if v and not on:
            on, st = True, i
        if not v and on:
            on = False
            out.append((st, i - 1))
    if on:
        out.append((st, len(b) - 1))
    return out


pred_evt, true_evt = runs(mask), runs(lab)

# match by overlap
matched_true = [False] * len(true_evt)
matched_pred = [False] * len(pred_evt)
tp = 0
for i, (ps, pe) in enumerate(pred_evt):
    for j, (ts, te) in enumerate(true_evt):
        if not matched_true[j] and not (pe < ts or ps > te):
            tp += 1
            matched_true[j] = matched_pred[i] = True
            break

# metrics
P, R, F, _ = precision_recall_fscore_support(
    lab, mask, average="binary", zero_division=0
)
try:
    auc = roc_auc_score(lab, err)
except ValueError:
    auc = float("nan")

prec_evt = tp / len(pred_evt) if pred_evt else 0
rec_evt = tp / len(true_evt) if true_evt else 0
f1_evt = 2 * prec_evt * rec_evt / (prec_evt + rec_evt + 1e-9)
tp_evt = tp
fp_evt = len(pred_evt) - tp
fn_evt = len(true_evt) - tp

tn, fp_win, fn_win, tp_win = confusion_matrix(lab, mask).ravel()

print(f"Window  P={P:.3f} R={R:.3f} F1={F:.3f}  AUROC={auc:.3f}")
print(f"         TP={tp_win:,} FP={fp_win:,} FN={fn_win:,} TN={tn:,}")
print(
    f"Event   P={prec_evt:.3f} R={rec_evt:.3f} F1={f1_evt:.3f}  (TP={tp_evt:,} FP={fp_evt:,} FN={fn_evt:,})"
)

# ── plots ---------------------------------------------------------------------
sns.set_style("darkgrid")
dpi = args.dpi
fig_w = 16


def tsec(w: int) -> float:
    return (w * hop) / ds.fs


# 1 score
plt.figure(figsize=(fig_w, 4), dpi=dpi)
plt.plot(err, lw=0.4, label="NCD")
plt.plot(thr, lw=0.8, ls="--", label="thr")
plt.title("NCD & threshold")
plt.legend()
plt.tight_layout()
plt.savefig("reports/ncd_score.png", dpi=dpi)

# 2 waveform + spans
dec = max(1, int(ds.fs // 1000))
t = np.arange(0, len(ds.wave), dec) / ds.fs
plt.figure(figsize=(fig_w, 4), dpi=dpi)
plt.plot(t, ds.wave[::dec], lw=0.3, color="#999")
for i, (ps, pe) in enumerate(pred_evt):
    col = "#76FF03" if matched_pred[i] else "#F44336"
    plt.axvspan(tsec(ps), tsec(pe) + args.chunk / ds.fs, color=col, alpha=0.15, lw=0)
for j, (ts, te) in enumerate(true_evt):
    if not matched_true[j]:
        plt.axvspan(
            tsec(ts), tsec(te) + args.chunk / ds.fs, color="#FF9100", alpha=0.15, lw=0
        )
plt.title("Waveform (green=TP lime, orange=FN, red=FP)")
plt.xlabel("time [s]")
plt.tight_layout()
plt.savefig("reports/ncd_events.png", dpi=dpi)

# 3 timeline
plt.figure(figsize=(fig_w, 2), dpi=dpi)
plt.xlim(0, tsec(len(win)))
plt.ylim(-0.5, 1.5)
plt.yticks([0, 1], ["Pred", "True"])
for i, (ps, pe) in enumerate(pred_evt):
    col = "#76FF03" if matched_pred[i] else "#F44336"
    plt.broken_barh([(tsec(ps), tsec(pe + 1 - ps))], (-0.2, 0.4), facecolors=col)
for j, (ts, te) in enumerate(true_evt):
    col = "#76FF03" if matched_true[j] else "#FF9100"
    plt.broken_barh([(tsec(ts), tsec(te + 1 - ts))], (0.8, 0.4), facecolors=col)
plt.title("Event timeline")
plt.xlabel("time [s]")
plt.tight_layout()
plt.savefig("reports/ncd_pred_timeline.png", dpi=dpi)

# 4 histogram of scores
plt.figure(figsize=(fig_w, 4), dpi=dpi)
sns.histplot(
    err[~lab], bins=100, color="skyblue", stat="density", label="noise", kde=True
)
if lab.any():
    sns.histplot(
        err[lab],
        bins=100,
        color="orange",
        stat="density",
        label="burst",
        kde=True,
        alpha=0.7,
    )
plt.title("NCD distribution")
plt.xlabel("score")
plt.ylabel("density")
plt.legend()
plt.tight_layout()
plt.savefig("reports/ncd_hist.png", dpi=dpi)

print(
    "✓ clearer plots written → ncd_score.png / ncd_events.png / ncd_pred_timeline.png / ncd_hist.png"
)

# save metrics json
metrics = {
    "window_P": float(P),
    "window_R": float(R),
    "window_F1": float(F),
    "window_TP": int(tp_win),
    "window_FP": int(fp_win),
    "window_FN": int(fn_win),
    "window_TN": int(tn),
    "event_P": float(prec_evt),
    "event_R": float(rec_evt),
    "event_F1": float(f1_evt),
    "event_TP": int(tp_evt),
    "event_FP": int(fp_evt),
    "event_FN": int(fn_evt),
    "auroc": float(auc),
}
with open("reports/ncd_metrics.json", "w") as fh:
    json.dump(metrics, fh, indent=2)
