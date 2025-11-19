# Image Content Moderation Model

## 1. Model Overview

M1: High-level purpose of the model
This model classifies user-uploaded images into policy categories
(e.g. safe, adult, violent, self-harm) to support **content moderation workflows**.

M2: Architecture
The model is a CNN-based encoder (EfficientNet-B4) with a multi-label
classification head trained on internal moderation labels.

M3: Stakeholders
- Trust & Safety team
- Policy and Legal
- Product moderation tooling team
- External auditors for safety reviews

## 2. Model Usage

M4: Intended use
The model is intended to:
- Pre-screen uploaded images
- Provide suggested labels to human moderators
- Trigger automated blocks for **highest risk** categories (e.g. CSAM hash hits)

M5: Prohibited uses
The model should **not** be used for:
- Law-enforcement or criminal investigation as primary evidence
- Identifying individuals or protected attributes
- Automated suspension decisions without human review

M6: Human-in-the-loop
All high-impact decisions (account suspension, law-enforcement referral)
require a human moderator review. The UI shows:
- Model labels and confidence scores
- Explanatory overlays (saliency maps)

## 3. Data

M7: Training data sources
We use a mixture of:
- Internally labeled user content (with consent and policy restrictions)
- Publicly available datasets (e.g. OpenImages variants)
- Vendor datasets for sensitive categories

M8: Label taxonomy
Top-level labels:

| Label ID | Description           |
|---------|------------------------|
| SAFE    | No policy violation    |
| ADULT   | Nudity / sexual content|
| VIOL    | Violence / gore        |
| SH      | Self-harm imagery      |
| OTH     | Other policy issues    |

M9: Privacy and security
- All images are stored in encrypted object storage.
- Access is limited to the moderation and ML teams.
- Raw data containing personally identifiable information is not shared
  outside the company.

## 4. Training

M10: Training regime
We train using mini-batch SGD with momentum:

- Batch size: 64
- Learning rate schedule: cosine decay from 1e-3 to 1e-6
- Weight decay: 1e-4
- Loss: weighted binary cross-entropy to handle class imbalance

M11: Data augmentation
We apply:
- Random crops and flips
- Color jitter
- Mild blurring
- Mixup with α = 0.2 for robustness

M12: Compute & tooling
- Training cluster: 8 × A100 GPUs
- Framework: PyTorch 2.x
- Experiment tracking: `mlflow` with automatic logging

## 5. Evaluation

M13: Metrics
For each label we compute:
- Precision / recall / F1
- ROC AUC
- Calibration (reliability curves)

Illustrative results on a held-out test set:

| Label | Precision | Recall | F1   | ROC AUC |
|-------|-----------|--------|------|---------|
| SAFE  | 0.96      | 0.94   | 0.95 | 0.99    |
| ADULT | 0.91      | 0.88   | 0.90 | 0.97    |
| VIOL  | 0.89      | 0.85   | 0.87 | 0.96    |
| SH    | 0.86      | 0.80   | 0.83 | 0.95    |

M14: Threshold selection
We tune thresholds using the following objective:
$$
\max_{\tau} \ \text{F}_\beta(\tau) \quad \text{with } \beta = 0.5
$$
to favour precision over recall for high-risk labels.

M15: Stress testing
We conduct stress tests on:
- Adversarial perturbations (low-strength)
- Distribution shift (new content sources)
- Synthetic combinations (collages, memes)

## 6. Monitoring

M16: Live monitoring
Daily dashboards track:
- Label frequency over time
- Drift in embedding space (cosine distance vs training distribution)
- Moderator disagreement rates

M17: Alerting
Alerts trigger when:
- Disagreement between model and moderators exceeds 10% over 7 days
- Drift metrics exceed pre-defined thresholds
- Volume of very high-risk content spikes

## 7. Risks

M18: High-level risk summary
Risks include:
- Failure to detect harmful content (false negatives)
- Over-blocking benign content (false positives)
- Unintended bias against particular styles, cultures, or subcultures

M19: Risk table

| Risk ID | Description                        | Likelihood | Impact | Mitigation                                      |
|---------|------------------------------------|------------|--------|-------------------------------------------------|
| R1      | Missed harmful but subtle content  | Medium     | High   | Human review, escalation paths, continuous data |
| R2      | Over-blocking artistic content     | Medium     | Medium | Policy review, exemption workflows              |
| R3      | Bias against certain demographics  | Low        | High   | Fairness audits, diverse data, external review  |

## 8. Governance

M20: Governance and audit
- Model owner: Head of Trust & Safety ML
- Policy owner: Head of Policy
- Audit trail:
  - All model versions stored in registry
  - All changes to thresholds and routing rules logged
  - Quarterly review with cross-functional stakeholders
