# Credit Risk Probability-of-Default Model

## 1. Model Overview

M1: High-level purpose of the model
This model estimates the **12-month probability of default (PD)** for retail credit customers
based on their application data, behavioural history, and bureau information.

M2: Business context
The model is used within the credit risk stack of the retail banking division to support:
- Credit limit decisions
- Pricing decisions
- Regulatory capital estimation (IRB slotting)

M3: Stakeholders
Primary stakeholders are:
- Credit risk team
- Retail product owners
- Model risk management
- Regulators (PRA / ECB)

## 2. Model Usage

M4: Intended use
The PD estimate is used as an **input** to:
- Expected loss calculation: `EL = PD × LGD × EAD`
- Risk-based pricing engines
- Regulatory capital models under IRB

M5: Out-of-scope uses
The PD estimate **must not** be used:
- As a stand-alone decision for automatic declines without human review
- For marketing segmentation outside of credit eligibility
- For non-customers (prospects without a credit file)

M6: User interactions
Front-office systems receive PD estimates through an API.
Users see:
- PD band (Very Low, Low, Medium, High, Very High)
- Key drivers explanation (top 3 features)

## 3. Data

M7: Data sources
The following data sources are used:

| Source              | Type             | Description                          |
|---------------------|------------------|--------------------------------------|
| Application system  | Structured table | Demographics, income, employment     |
| Core banking        | Structured table | Account balances, arrears, write-offs|
| Credit bureau       | External feed    | Delinquencies, inquiries, trades     |

M8: Data coverage and representativeness
Training data covers **Jan 2015 – Dec 2023**, EU retail customers only.
Known gaps:
- Limited representation of customers under 21 years old
- No data from non-EU branches
- Sparse history for new products launched in 2022

M9: Data quality and preprocessing
Key preprocessing steps:
1. Impute missing income using country × employment segment medians.
2. Cap extreme values at the 0.5% / 99.5% quantiles.
3. One-hot encode categorical variables with frequency threshold 0.5%.
4. Standardise numeric features to zero mean / unit variance.

## 4. Training

M10: Training procedure
The model is a gradient boosted tree (`XGBoost`) fitted on a balanced dataset.
Objective: `binary:logistic`.

Key hyperparameters:

| Parameter | Value |
|-----------|-------|
| max_depth | 5     |
| eta       | 0.05  |
| n_rounds  | 450   |
| subsample | 0.8   |

M11: Target definition
Default is defined as:
- 90+ days past due, OR
- Write-off, OR
- Forborne restructuring with financial hardship

Observation window: 12 months from origination.
Examples with censoring < 12 months are excluded.

M12: Regularisation and constraints
To reduce overfitting:
- Early stopping on validation AUC with patience = 50 rounds
- L2 regularisation `lambda = 2`
- Minimum child weight = 10

## 5. Evaluation

M13: Metrics and performance
Primary metrics:
- AUC on out-of-time test (2023 only)
- Brier score
- KS statistic

Results:

| Segment     | AUC  | KS   | Brier |
|-------------|------|------|-------|
| All         | 0.83 | 0.54 | 0.094 |
| Mortgages   | 0.80 | 0.49 | 0.089 |
| Credit card | 0.85 | 0.57 | 0.101 |

The Brier score is:
$$
\text{Brier} = \frac{1}{N} \sum_i (p_i - y_i)^2
$$

M14: Calibration
We apply:
- Platt scaling on validation data
- Annual back-testing of observed vs predicted default rates by PD band

Max absolute calibration error by band: 1.8 percentage points.

M15: Fairness analysis
We compare performance by age band, gender proxy, and region.

Findings (illustrative):
- No material difference in AUC across age bands.
- Slightly higher PDs for younger customers due to thinner credit files.
- No direct use of protected attributes; only business-justified proxies.

## 6. Monitoring

M16: Monitoring strategy
Daily monitoring:
- Volume and PD distribution by product
- Drift statistics on top 20 features

Monthly monitoring:
- Realised default rates vs predicted PDs by band
- Stability index (PSI) for key populations

Alert thresholds:
- PSI > 0.2 triggers review
- Absolute calibration error > 3 percentage points triggers deep dive

M17: Retraining policy
The model is retrained at least annually or when:
- Significant product changes occur
- Data drift exceeds thresholds for 3 consecutive months

## 7. Risks

M18: High-level risk summary
Key risks include:
- Mis-calibration leading to underestimation of portfolio risk
- Data drift under new macroeconomic regimes
- Unintended bias against under-represented groups

M19: **Detailed risk table**

| Risk ID | Description                               | Likelihood | Impact | Mitigation                              |
|---------|-------------------------------------------|------------|--------|-----------------------------------------|
| R1      | Underestimation of PD in downturn         | Medium     | High   | Periodic stress testing & overlays      |
| R2      | Data feed outage from bureau              | Low        | High   | Fallback scorecard + default overrides  |
| R3      | Model misinterpretation by front-office   | Medium     | Medium | Training, clear documentation, tooltips |

## 8. Governance

M20: Governance and approvals
- Model owner: Head of Retail Credit Risk
- Model sponsor: Retail Banking ExCo member
- Approvals required:
  - Model Risk Committee
  - Regulatory notification where applicable

Versioning:
- Major versions (v1, v2, …) require full validation.
- Minor versions (v1.1, v1.2, …) can follow a light-touch process.
