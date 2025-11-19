# Example Model Card

## 1. Model Overview

M1: High-level purpose of the model  
This model predicts the probability of a horse winning a race based on past performance, trainer, jockey, and in-play features.

M2: Key audiences  
- Quant researchers  
- Risk managers  
- Regulators  

Here is a small table summarising versions:

| Version | Date       | Notes              |
|--------|------------|--------------------|
| v1.0   | 2025-01-10 | Initial deployment |
| v1.1   | 2025-03-05 | Added new features |

M3: Known limitations  
1. Sparse data for rare race conditions.  
2. Potential data drift over time.  

## 2. Model Usage

M4: Intended use  
The model is intended for **internal risk monitoring** only.  
It is *not* designed for consumer-facing betting advice.

M5: Out-of-scope uses  
- Using outputs as guaranteed edge in trading  
- Marketing claims of certain profit  

## 3. Evaluation

M6: Metrics and performance  
The primary metric is AUC.  
On the held-out test set we obtained AUC = 0.78.

We also used a calibration test with Brier score:

$$
\text{Brier} = \frac{1}{N}\sum_i (p_i - y_i)^2
$$

M7: Fairness considerations  
We compared performance by race class and track type.  
No statistically significant differences were found, but further study is planned.

## 4. Risks

M8: High-level risk summary  
There is a risk of over-reliance on the model under unusual market conditions.

M9: Detailed risk table  

| Risk ID | Description                           | Stage | Status  |
|--------|----------------------------------------|-------|---------|
| R1     | Model performance under data drift     | Live  | Open    |
| R2     | Misinterpretation of probabilities     | Live  | Mitigated |
