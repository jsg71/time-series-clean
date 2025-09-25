#!/usr/bin/env python

"""
─────────────────────────────────────────────────────────────────────────────
 sim_make.py — Convenience wrapper around `simulate()` to create synthetic
               “field recordings” + alias file used by training / eval
 Author : jsg
─────────────────────────────────────────────────────────────────────────────

WHAT THE SCRIPT DOES
────────────────────
1. Calls `leela_ml.signal_sim.simulator.simulate(minutes, out, seed)` — a
   comprehensive signal generator that writes:
       • <out>_<hash>.npy     floating‑point waveform
       • <out>_<hash>.json    metadata: sample‑rate, burst timestamps, RNG seed
2. Finds the *first* generated *.npy* and **copies** it to
   `<out>_wave.npy`.
   Down‑stream scripts (`train_ae*.py`, `eval_ae*.py`, `run_ncd.py`) accept
   this alias so you don’t have to remember the long hash file‑name.

CLI ARGUMENTS
─────────────
--minutes  INT  default 5
      Length of simulated recording in *minutes of real time*.
      At 40 kHz that’s 60 s × 5 × 40 000 ≈ 12 000 000 samples.

--out  PATH  required
      Prefix for output files.  Example `data/stationA` →
      • data/stationA_<hash>.npy
      • data/stationA_<hash>.json
      • data/stationA_wave.npy  (alias copy)

--seed INT  default 0
      RNG seed forwarded to `simulate()` for repeatability.
      Using the same seed reproduces identical waveform + burst timings.

REALISM OF GENERATED DATA
─────────────────────────
The underlying `simulate()` routine (see package docs) aims to mimic:

• **Background noise floor** – pink noise + sensor‑specific white noise  
• **Lightning bursts** – Gaussian‑shaped envelope, fundamental + harmonics  
• **Drift** – slow ±2 dB gain drift over minutes (sim hardware warming)  
• **ADC saturation** – clips peaks beyond ±0.95 of dynamic range

Thus models trained on `*_wave.npy` experience artefacts similar to real
field recordings: varying burst amplitude, overlap with noise, occasional
clipping, and progressively changing noise statistics.

STRENGTHS
──────────
✓ One‑liner dataset generation; handy for unit tests & benchmarks  
✓ Deterministic via `--seed` → reproducible results, CI‑friendly  
✓ Produces both *continuous waveform* and *structured meta* JSON so
  downstream evaluators know the ground‑truth burst positions

WEAKNESSES & IMPROVEMENTS
─────────────────────────
✗ Only `--minutes` is exposed; other realism knobs (SNR, burst rate,
  clipping threshold) hard‑coded inside `simulate()`.  
✗ Uses `shutil.copy` which duplicates the waveform on disk; for multiple
  long recordings disk usage grows 2×.  *Fix*: create a symlink instead.  
✗ Hash picking `sorted(glob(...))[0]` assumes simulator writes only one
  pair; may grab the wrong file if directory already contained older runs.  
✗ No explicit CLI for sample‑rate; tied to simulator default (40 kHz).

IDEAS FOR FUTURE WORK
─────────────────────
• Expose `--rate`, `--snr_db`, `--burst_density` flags; forward to
  `simulate()` to vary realism across sweeps.  
• Add `--symlink_alias` boolean to create symlink instead of copy.  
• Provide `--meta_out` to override JSON path/location.  
• Emit log of parameters to *out.log* for provenance tracking.

USAGE EXAMPLE
─────────────
```bash
python sim_make.py \
    --minutes 7 \
    --out data/stationB \
    --seed 123
```
Creates:
    data/stationB_a9c7ef.npy
    data/stationB_a9c7ef.json
    data/stationB_wave.npy      # alias for convenience

The alias lets you later do:
    python train_ae_baseline.py --npy data/stationB_wave.npy --meta data/stationB_meta.json ...
without hunting for hashed filenames.

─────────────────────────────────────────────────────────────────────────────
"""  # end of header


import argparse, shutil, glob
from leela_ml.signal_sim.simulator import simulate

p=argparse.ArgumentParser()
p.add_argument("--minutes",type=int,default=5)
p.add_argument("--out",required=True)
p.add_argument("--seed",type=int,default=0)
a=p.parse_args()

simulate(a.minutes,a.out,seed=a.seed)
first=sorted(glob.glob(f"{a.out}_*.npy"))[0]
shutil.copy(first,f"{a.out}_wave.npy")
print("alias →",f"{a.out}_wave.npy")
