---
tags: [demo, autoencoder, resnet]
license: MIT
library_name: pytorch
language: en
---

# Model Card — ResNet Autoencoder (demo)

## Model Details
- **Owner**: Your Team
- **Task**: Reconstruction-based anomaly detection on waveform windows
- **Backbone**: 1D ResNet encoder/decoder (skip connections)
- **Objective**: L1 / L2 reconstruction loss
- **Status**: Template (wire in your class/module when ready)
- **Primary module**: *(fill in, e.g., `leela_ml.models.resnet_ae.ResnetAE`)*

## Intended Use
- **Intended**: Feature-rich baseline to flag structural deviations in fixed-length windows.
- **Out of scope**: Causal inference, root-cause analysis, safety-critical decisions.

## Architecture (example)
- **Encoder**: [Conv → BN → ReLU] × *n* with residual blocks; downsampling via stride.
- **Bottleneck**: latent dimension *z* (e.g., 64–256).
- **Decoder**: symmetric upsampling (ConvTranspose1d or interpolation + Conv).
- **Normalization**: per-window or per-channel standardization.

## Training & Hyperparameters
- **Loss**: `L = mean(|x - x_hat|)` (L1) or `mean((x - x_hat)^2)` (L2)
- **Optimizer**: Adam (lr=1e-3), weight decay 1e-5
- **Regularization**: dropout(0.1–0.3), early stopping on val loss
- **Batch size**: 64–256; **epochs**: 20–100
- **Augmentations**: jitter, time-shift, small amplitude scaling (optional)

## Evaluation
- **Score**: residual per window, e.g. `||x - x_hat||_1 / L`
- **Metrics**: PR-AUC (for imbalance), ROC-AUC
- **Operating point**: threshold at fixed FPR (0.1–1%) or via val quantile

## Risks & Limitations
- Can **reconstruct anomalies** (false negatives) when capacity is high.
- Windowing may **miss short events**; consider multi-scale windows.
- Sensitive to normalization drift across deployments.

## Usage (pseudo)
```python
# Replace with your actual class once implemented
from leela_ml.models.resnet_ae import ResnetAE  # ← your module
m = ResnetAE(channels=1, depth=4, bottleneck=128)
m.fit(train_windows, epochs=50)
xhat = m.reconstruct(test_windows)
resid = (test_windows - xhat).abs().mean(dim=-1)
labels = resid > tau  # choose tau on validation
```

