# Insurance Claims Prediction — Canadian P&C Market

**Author:** Reda Hakkani | PhD Candidate, Applied Mathematics | Montréal, QC  
**Domain:** Machine Learning · Actuarial Pricing · Canadian P&C Insurance  
**Regulatory context:** FSRA (ON) · AMF (QC) · AIRB (AB) · OSFI E-23 · IBC CANATICS

---

## Overview

Full end-to-end ML pipeline for **binary claims prediction** on Canadian personal auto and commercial lines.  
Calibrated to IBC industry statistics across 6 provinces: Ontario, Quebec, Alberta, BC, Manitoba, Saskatchewan.

**Pipeline:** EDA → Feature Engineering → GLM Baseline → XGBoost → Isotonic Calibration → SHAP

---

## Results

| Metric | GLM Baseline | XGBoost | Calibrated XGBoost |
|--------|-------------|---------|-------------------|
| AUC-ROC | 0.569 | 0.563 | **0.641** |
| Gini Normalized | 0.138 | 0.126 | **0.282** |
| Decile 1 Lift | 1.3x | 1.5x | **3.1x** |

---

## Canadian Market Features

### Dataset — 595,212 Policies
| Feature | Description | Regulatory Relevance |
|---------|-------------|---------------------|
| `province` | ON / QC / AB / BC / MB / SK | Territory rating (FSRA/AMF) |
| `vehicle_age` | 0–3 categories | IBC actuarial tables |
| `driver_age_grp` | 5 age bands | FSRA rating factor |
| `prior_claims` | Claims history | Experience rating |
| `canatics_flag` | IBC fraud network flag | CANATICS integration |
| `avg_daily_km` | Telematics UBI | Intact / Desjardins PAYD |
| `night_driving` | % nocturnal | UBI risk indicator |
| `hard_braking` | Events / month | Telematics scoring |
| `at_fault` | Fault determination | AB / ON direct compensation |

### Engineered Features
```python
'young_at_fault'         # High-risk interaction (driver_age_grp=1 × at_fault)
'fraud_signal'           # CANATICS flag × no witnesses
'high_risk_telematics'   # Hard braking OR speeding composite
'repeat_claimant'        # Prior claims ≥ 2
'cost_premium_ratio'     # Total claim cost / annual premium
'night_hard_interaction' # Night driving × hard braking events
```

---

## Pipeline Architecture

```
595,212 Canadian P&C Policies
           │
           ▼
    ┌─────────────────────┐
    │   EDA & Quality     │
    │ - Province breakdown│
    │ - Claim rate: 5.1%  │
    │ - Class imbalance   │
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │ Feature Engineering │
    │ 21 → 31 features    │
    │ UBI + CANATICS +    │
    │ Interaction terms   │
    └──────────┬──────────┘
               │
         ┌─────┴──────┐
         ▼            ▼
    ┌─────────┐  ┌───────────────┐
    │   GLM   │  │  XGBoost      │
    │Baseline │  │ 250 estimators│
    │AUC:0.569│  │ AUC: 0.563    │
    └─────────┘  └──────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │   Isotonic      │
               │  Calibration    │
               │  AUC: 0.641     │
               └────────┬────────┘
                        │
                        ▼
               ┌─────────────────┐
               │  SHAP Values    │
               │ OSFI E-23 ready │
               │ Underwriting UI │
               └─────────────────┘
```

---

## Claim Rate by Province

| Province | Policies | Claim Rate |
|----------|----------|------------|
| Ontario | 226,110 | 5.10% |
| Quebec | 142,841 | 5.29% |
| Alberta | 95,617 | 5.26% |
| British Columbia | 83,104 | 4.76% |
| Manitoba | 23,743 | 5.08% |
| Saskatchewan | 11,875 | 5.15% |

---

## Top Predictors (Feature Importance)

| Rank | Feature | Importance | Canadian Context |
|------|---------|------------|-----------------|
| 1 | `vehicle_age` | 0.115 | IBC actuarial classification |
| 2 | `night_driving` | 0.110 | UBI / PAYD telematics |
| 3 | `prior_claims` | 0.092 | Experience rating (FSRA) |
| 4 | `night_hard_interaction` | 0.091 | Composite risk indicator |
| 5 | `repair_cost` | 0.075 | AB/ON direct compensation |
| 6 | `speeding_pct` | 0.056 | Telematics UBI |

---

## Regulatory Alignment

| Standard | Application | Status |
|----------|-------------|--------|
| **FSRA (ON)** | Rate filing support | ✓ Territory + experience factors |
| **AMF (QC)** | Tarification auto | ✓ Territory pricing |
| **AIRB (AB)** | Grid rating system | ✓ Vehicle class alignment |
| **OSFI E-23** | Model risk management | ✓ SHAP interpretability |
| **IBC CANATICS** | Fraud detection network | ✓ Flag integrated |

---

## Installation

```bash
git clone https://github.com/RedaHakkani/insurance-claims-prediction.git
cd insurance-claims-prediction
pip install -r requirements.txt
python src/claims_prediction.py
```

## Requirements

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
```

## Output

- Full console report (province breakdown, lift table, feature importance)
- `claims_prediction_results.png` — 6-panel dashboard

---

## References

- IBC (2023). *Facts of the General Insurance Industry in Canada*.
- FSRA (2022). *Ontario Automobile Insurance — Rate Filing Guidelines*.
- OSFI (2017). *Guideline E-23 — Model Risk Management*.
- AMF (2023). *Lignes directrices — Tarification de l'assurance automobile*.

---

*Reda Hakkani — PhD Candidate, Applied Mathematics | Montréal, QC*  
*Available for actuarial and quantitative risk roles — hakkanireda@hotmail.com*
