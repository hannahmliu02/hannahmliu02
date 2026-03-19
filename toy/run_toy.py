"""
run_toy.py
----------
Single-entry-point orchestration script.

Usage:
    python -m toy.run_toy

Pipeline:
  1. generate_data  — synthetic patient trajectories with biased observation
  2. train_model    — logistic regression risk model on engineered features
  3a. audit          — 1-step counterfactual masking observability audit
  3b. multistep      — multi-step masking: risk decay over k consecutive
                       unmeasured steps (actual start + controlled start)
  4. simulate       — score-based monitoring policy (baseline)
  5. mitigation     — uncertainty-triggered measurement + comparison

All outputs (plots + CSVs + JSON) are written to outputs/.
A short disparity table is printed to stdout.
"""

import json
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------ #
# Shared parameters
# ------------------------------------------------------------------ #
PARAMS = dict(
    # Data generation
    N=2000,
    T=20,
    p11=0.8,
    p01=0.2,
    meas_prob_s0=(0.30, 0.15),   # P(measure | S=0) for [group0, group1]
    meas_prob_s1=(0.80, 0.60),   # P(measure | S=1) for [group0, group1]
    mu0=0.0,
    mu1=1.0,
    sigma=1.0,
    noise_flip=0.05,
    seed=42,
    # Policy simulation
    T_sim=30,
    tau=0.5,
    p_high=0.85,
    p_low=0.15,
    sim_seed=123,
    # Mitigation thresholds
    u0=0.20,
    g0=3,
)

OUT_DIR = "outputs"


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- #
    # Step 1 — Generate data
    # ---------------------------------------------------------------- #
    print("=" * 62)
    print("[1/5] Generating synthetic patient trajectories ...")
    from toy.generate_data import generate_data

    df = generate_data(
        N=PARAMS["N"],
        T=PARAMS["T"],
        p11=PARAMS["p11"],
        p01=PARAMS["p01"],
        meas_prob_s0=PARAMS["meas_prob_s0"],
        meas_prob_s1=PARAMS["meas_prob_s1"],
        mu0=PARAMS["mu0"],
        mu1=PARAMS["mu1"],
        sigma=PARAMS["sigma"],
        noise_flip=PARAMS["noise_flip"],
        seed=PARAMS["seed"],
    )
    print(f"   {len(df)} rows  ({PARAMS['N']} patients × {PARAMS['T']} steps)")
    meas_by_g = df.groupby("group")["M"].mean()
    sev_by_g  = df.groupby("group")["S"].mean()
    print(f"   Measurement rate — group0: {meas_by_g[0]:.3f}  "
          f"group1: {meas_by_g[1]:.3f}  "
          f"(injected disparity: {meas_by_g[0]-meas_by_g[1]:.3f})")
    print(f"   True severity   — group0: {sev_by_g[0]:.3f}  "
          f"group1: {sev_by_g[1]:.3f}")

    # ---------------------------------------------------------------- #
    # Step 2 — Train model
    # ---------------------------------------------------------------- #
    print("\n[2/5] Building features and training risk model ...")
    from toy.train_model import build_features, train_model, save_model

    df_feat = build_features(df)
    model, metrics, df_test = train_model(df_feat, seed=PARAMS["seed"])
    save_model(model, f"{OUT_DIR}/model.pkl")

    print(f"   AUROC: {metrics['auroc']:.4f}")

    # ---- Wald significance table (statsmodels) --------------------------
    sig_df = pd.DataFrame(metrics["coef_significance"])
    sig_df.to_csv(f"{OUT_DIR}/model_coef_significance.csv", index=False)

    print("   Logistic regression coefficients  (statsmodels Wald test, no regularisation):")
    print(f"   {'Feature':<15} {'Coef':>8} {'SE':>7} {'z':>7} {'p-value':>10} "
          f"{'95% CI':>18} {'Sig':>4}")
    print(f"   {'-'*73}")
    shortcut_feats = {"missing_now", "gap"}
    for _, row in sig_df.iterrows():
        ci_str = f"[{row['ci_025']:+.3f}, {row['ci_975']:+.3f}]"
        flag   = " ←" if row["feature"] in shortcut_feats else ""
        print(f"   {row['feature']:<15} {row['coef']:>+8.4f} {row['std_err']:>7.4f} "
              f"{row['z']:>7.3f} {row['pvalue']:>10.4f} {ci_str:>18} "
              f"{row['stars']:>4}{flag}")
    print(f"   Significance: *** p<0.001  ** p<0.01  * p<0.05  . p<0.10  ns p≥0.10")

    # Verify: model predicts lower risk for group 1 in test set
    pred_by_g = df_test.groupby("group")["pred_prob"].mean()
    print(f"   Mean predicted risk — group0: {pred_by_g[0]:.3f}  "
          f"group1: {pred_by_g[1]:.3f}  "
          f"(diff: {pred_by_g[0]-pred_by_g[1]:+.3f})")

    with open(f"{OUT_DIR}/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ---------------------------------------------------------------- #
    # Step 3 — Observability audit
    # ---------------------------------------------------------------- #
    print("\n[3/5] Running observability audit (counterfactual masking) ...")
    from toy.audit import run_audit

    audit_summary = run_audit(model, df_test, out_dir=OUT_DIR)
    print(f"   Mean Δrisk = risk(observed) − risk(masked):")
    print(f"     Group 0: {audit_summary['group0_mean_delta']:+.4f}")
    print(f"     Group 1: {audit_summary['group1_mean_delta']:+.4f}")
    print(f"   Plots: audit_delta_risk_hist.png  audit_delta_risk_vs_gap.png")

    # ---------------------------------------------------------------- #
    # Step 3b — Multi-step masking audit
    # ---------------------------------------------------------------- #
    print("\n[3b/5] Multi-step masking audit ...")
    from toy.audit import run_multistep_masking

    highlight_k = (3, 5, 10)
    ms_agg = run_multistep_masking(
        model, df_feat,
        K_max=10,
        highlight_k=highlight_k,
        out_dir=OUT_DIR,
    )

    print(f"   Predicted risk at key checkpoints (actual historical start):")
    print(f"   {'k':>4}  {'Group0 risk':>12}  {'Group1 risk':>12}  {'G0−G1 gap':>10}")
    print(f"   {'-'*44}")
    for kv in (0,) + highlight_k:
        row0 = ms_agg[(ms_agg["k"] == kv) & (ms_agg["group"] == 0)]
        row1 = ms_agg[(ms_agg["k"] == kv) & (ms_agg["group"] == 1)]
        r0 = row0["mean_risk"].values[0]
        r1 = row1["mean_risk"].values[0]
        print(f"   {kv:>4}  {r0:>12.4f}  {r1:>12.4f}  {r0-r1:>10.4f}")
    print(f"   Plots: audit_multistep_masking.png  audit_multistep_controlled.png")

    # ---------------------------------------------------------------- #
    # Steps 4 & 5 — Baseline + mitigated simulation
    # ---------------------------------------------------------------- #
    print("\n[4-5/6] Simulating score-based deployment policy ...")
    from toy.mitigation import compare_simulations

    sim_kwargs = dict(
        T_sim   = PARAMS["T_sim"],
        p11     = PARAMS["p11"],
        p01     = PARAMS["p01"],
        mu0     = PARAMS["mu0"],
        mu1     = PARAMS["mu1"],
        sigma   = PARAMS["sigma"],
        tau     = PARAMS["tau"],
        p_high  = PARAMS["p_high"],
        p_low   = PARAMS["p_low"],
        seed    = PARAMS["sim_seed"],
    )

    mit_summary = compare_simulations(
        model     = model,
        df_feat   = df_feat,
        sim_kwargs= sim_kwargs,
        u0        = PARAMS["u0"],
        g0        = PARAMS["g0"],
        p_high    = PARAMS["p_high"],
        out_dir   = OUT_DIR,
    )

    print(f"\n   Disparity = (Group-0 value) − (Group-1 value)")
    print(f"   Positive value → Group-1 is disadvantaged (under-measured / under-estimated).")
    print(f"\n   Legend: disparity = (G0 value) − (G1 value).")
    print(f"   For Measurement Rate and Predicted Risk: positive = G1 under-served.")
    print(f"   For Gap Length: negative = G1 has LONGER gap (i.e. under-served).")
    print(f"   Improvement = reduction in |disparity|.")
    print(f"\n   {'Metric':<30} {'Baseline':>12} {'Mitigated':>12} {'|Δ| improved':>14}")
    print(f"   {'-'*70}")
    for key, label in [("meas", "Measurement Rate"),
                        ("risk", "Predicted Risk"),
                        ("gap",  "Gap Length (G1 longer→neg)")]:
        base = mit_summary[f"baseline_{key}_disparity"]
        mit  = mit_summary[f"mitigated_{key}_disparity"]
        imp  = abs(base) - abs(mit)   # positive = absolute disparity reduced
        marker = " ✓" if imp > 0.001 else ""
        print(f"   {label:<30} {base:>12.4f} {mit:>12.4f} {imp:>14.4f}{marker}")

    # ---------------------------------------------------------------- #
    # Step 6 — Causal RLHF mitigation
    # ---------------------------------------------------------------- #
    print("\n[6/6] Running Causal RLHF pipeline ...")
    print("   Graph:  t=1: x→M1→y1  |  t=2: M2→y2→RM1, h→M1→y1  |  t=3: y1→RM2→M2")
    from toy.causal_rlhf import run_causal_rlhf

    rlhf_summary = run_causal_rlhf(
        m1_model   = model,
        df_feat    = df_feat,
        sim_kwargs = sim_kwargs,
        n_rounds   = 8,
        out_dir    = OUT_DIR,
    )

    print(f"\n   Causal RLHF vs Baseline — Simulation Disparity")
    print(f"   (disparity = G0 value − G1 value; positive → G1 under-served)")
    print(f"\n   {'Metric':<30} {'Baseline':>12} {'Causal RLHF':>12} {'|Δ| improved':>14}")
    print(f"   {'-'*70}")
    for key, label in [("meas", "Measurement Rate"),
                        ("risk", "Predicted Risk"),
                        ("gap",  "Gap Length (G1 longer→neg)")]:
        base = rlhf_summary[f"causal_rlhf_baseline_{key}_disparity"]
        rlhf = rlhf_summary[f"causal_rlhf_{key}_disparity"]
        imp  = abs(base) - abs(rlhf)
        marker = " ✓" if imp > 0.001 else ""
        print(f"   {label:<30} {base:>12.4f} {rlhf:>12.4f} {imp:>14.4f}{marker}")

    print(f"\n   Causal RLHF Training (prediction-level):")
    print(f"     y1 (M1) adjusted disparity : "
          f"{rlhf_summary['causal_rlhf_final_y1_adj_disparity']:+.4f}")
    print(f"     y2 (M2) adjusted disparity : "
          f"{rlhf_summary['causal_rlhf_final_y2_adj_disparity']:+.4f}")
    print(f"     Final RM1 fairness score   : "
          f"{rlhf_summary['causal_rlhf_final_rm1_score']:.4f}")
    print(f"   Plots: causal_rlhf_training_curves.png  causal_rlhf_comparison.png")

    # ---------------------------------------------------------------- #
    # Save full experiment log
    # ---------------------------------------------------------------- #
    full_summary = {**PARAMS, **metrics, **audit_summary, **mit_summary, **rlhf_summary}
    for k, v in full_summary.items():
        if isinstance(v, tuple):
            full_summary[k] = list(v)
    with open(f"{OUT_DIR}/experiment_summary.json", "w") as f:
        json.dump(full_summary, f, indent=2)

    print(f"\n{'='*62}")
    print("All outputs written to  outputs/")
    print("  model_coef_significance.csv        Wald test table (statsmodels)")
    print("  audit_delta_risk_hist.png          1-step Δrisk histogram")
    print("  audit_delta_risk_vs_gap.png        Δrisk vs gap (binned)")
    print("  audit_summary.csv")
    print("  audit_multistep_masking.png        risk decay over k steps (actual state)")
    print("  audit_multistep_controlled.png     risk decay over k steps (gap reset to 0)")
    print("  audit_multistep_summary.csv")
    print("  sim_baseline.png                   baseline policy trajectories")
    print("  sim_mitigation.png                 mitigated policy trajectories")
    print("  mitigation_comparison.png          side-by-side comparison")
    print("  mitigation_summary.csv")
    print("  causal_rlhf_training_curves.png    RM1 score + disparity over rounds")
    print("  causal_rlhf_comparison.png         Causal RLHF vs baseline simulation")
    print("  causal_rlhf_training_history.csv   per-round pipeline metrics")
    print("  causal_rlhf_sim_summary.csv        simulation summary table")
    print("  model.pkl  |  model_metrics.json  |  experiment_summary.json")
    print("=" * 62)


if __name__ == "__main__":
    main()
