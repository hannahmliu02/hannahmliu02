# Bias Amplification in Clinical Risk Scoring: A Causal Analysis

Research project investigating how measurement bias in clinical data is learned and amplified by risk models, and evaluating mitigation strategies including a novel Causal RLHF pipeline.

**Hannah Liu** · MSc AI Applications & Innovation, Imperial College London

---

## Overview

Clinical risk models trained on electronic health records can learn spurious correlations between *being measured* and *being sick*. When a subgroup of patients is systematically under-measured (due to access disparities, cost, or implicit bias), the model learns "long gap / missing data → lower risk" — even when the unmeasured patients are equally or more severely ill. Deployed in a monitoring policy, this creates a feedback loop: lower predicted risk → fewer measurements → longer gaps → even lower predicted risk.

This project:
1. Formalises the bias mechanism using synthetic patient trajectories
2. Audits trained models via counterfactual masking
3. Evaluates two mitigation strategies: uncertainty-triggered measurement and a Causal RLHF pipeline
4. Compares all three conditions in a simulated deployment environment

---

## Repository Structure

```
.
├── toy/                        # Synthetic experiment pipeline
│   ├── run_toy.py              # Main orchestration script (run this)
│   ├── generate_data.py        # Synthetic patient trajectories with biased observation
│   ├── train_model.py          # Feature engineering and logistic regression training
│   ├── audit.py                # Counterfactual observability audits
│   ├── simulate_policy.py      # Score-based monitoring policy simulation
│   ├── mitigation.py           # Uncertainty-triggered measurement mitigation
│   ├── causal_rlhf.py          # Causal RLHF debiasing pipeline
│   └── plot_three_way.py       # Three-way comparison plots
├── notebooks/
│   └── bias_analysis.ipynb     # Interactive analysis notebook
├── outputs/                    # Generated plots, CSVs, and summary JSON
└── README.md
```

---

## Quickstart

```bash
# Install dependencies
pip install numpy pandas scikit-learn statsmodels matplotlib

# Run the full experiment pipeline
python -m toy.run_toy

# Generate the three-way comparison figure
python -m toy.plot_three_way
```

All outputs (plots, CSVs, JSON summaries) are written to `outputs/`.

---

## Experiment Pipeline

### 1. Data Generation (`generate_data.py`)

Simulates N=2,000 patients over T=20 timesteps with a hidden Markov severity state S ∈ {0, 1}. Measurements are:
- More likely when S=1 (clinicians tend to monitor sicker patients)
- Systematically less frequent for Group 1 patients at every severity level, injecting a measurement disparity of ~0.2

### 2. Model Training (`train_model.py`)

Engineers four features per (patient, timestep):
- `x_ffill` — last observed clinical value (forward-filled)
- `missing_now` — whether the current timestep has no measurement
- `gap` — steps since last measurement
- `meas_count` — cumulative measurements so far

Trains a logistic regression to predict patient outcome Y. The `missing_now` and `gap` features are the bias shortcuts — because being measured correlates with severity in the training data, the model assigns lower predicted risk to patients who were not recently measured.

### 3. Observability Audit (`audit.py`)

**1-step counterfactual masking:** For every test-set record, computes Δrisk = risk(observed) − risk(masked). A positive Δrisk means the model assigns higher risk when the patient is observed — which systematically disadvantages Group 1.

**Multi-step masking:** Simulates k consecutive unmeasured steps and tracks how predicted risk decays, comparing groups under (a) actual historical starting states and (b) a controlled reset where all gaps start at 0.

### 4. Baseline Policy Simulation (`simulate_policy.py`)

Deploys the trained model in a risk-driven monitoring policy for T=30 timesteps:
- If predicted risk > τ=0.5 → measure with probability p_high=0.85
- Otherwise → measure with probability p_low=0.15

Initialises from each patient's real end-of-training state so that historical disparities carry forward. Demonstrates the feedback loop in action.

### 5. Uncertainty-Triggered Mitigation (`mitigation.py`)

Breaks the feedback loop by also triggering measurements when:
- Model uncertainty (p×(1−p)) exceeds threshold u₀=0.20, **or**
- Gap since last measurement exceeds g₀=3 steps

This forces measurement of high-gap patients regardless of predicted risk.

### 6. Causal RLHF (`causal_rlhf.py`)

A three-timestep causal pipeline that trains a debiasing model M2 on top of the frozen base model M1:

```
t=1:  x  → M1 → y1          (biased predictions from frozen M1)
t=3:  y1 → RM2 → M2         (RM2 diagnoses bias, trains M2)
t=2:  M2 → y2 → RM1         (M2 produces debiased predictions; RM1 audits fairness)
      h  → M1 → y1          (confounders continue influencing M1)
```

Key design choices:
- **M1 is frozen** throughout (like a pre-trained base model in RLHF)
- **M2 does not receive group membership** as a feature — it learns to debias from observable proxies (gap, missing_now) rather than directly from the protected attribute
- **RM2** uses group labels (available at training time) to identify h-caused bias and produce counterfactual correction targets
- **Iterative refinement**: each round RM2 evaluates the current M2 output, not M1's output, enabling genuine round-over-round improvement

---

## Key Results

| Metric | Baseline | Uncertainty-Triggered | Causal RLHF |
|--------|----------|----------------------|-------------|
| Measurement Rate Disparity (G0−G1) | +0.20 | reduced | reduced |
| Predicted Risk Disparity (G0−G1) | +0.06 | partial | further reduced |
| Gap Length Disparity | −1.2 | reduced | reduced |

See `outputs/three_way_combined.png` for the full visualisation and `outputs/experiment_summary.json` for numeric results.

---

## Outputs

| File | Description |
|------|-------------|
| `audit_delta_risk_hist.png` | Δrisk distribution by group (1-step masking) |
| `audit_delta_risk_vs_gap.png` | Δrisk vs gap length by subgroup |
| `audit_multistep_masking.png` | Risk decay over k unmeasured steps (actual state) |
| `audit_multistep_controlled.png` | Risk decay over k steps (gap reset to 0) |
| `sim_baseline.png` | Baseline policy deployment trajectories |
| `sim_mitigation.png` | Mitigated policy deployment trajectories |
| `mitigation_comparison.png` | Baseline vs uncertainty-triggered side-by-side |
| `causal_rlhf_training_curves.png` | RM1 fairness score + disparity over training rounds |
| `causal_rlhf_comparison.png` | Causal RLHF vs baseline simulation |
| `three_way_combined.png` | All three conditions compared |
| `disparity_bar_chart.png` | Summary disparity bar chart |
| `model_coef_significance.csv` | Logistic regression Wald significance table |
| `experiment_summary.json` | Full numeric summary of all experiments |

---

## Contact

Hannah Liu · [hannah.liu25@imperial.ac.uk](mailto:hannah.liu25@imperial.ac.uk)
