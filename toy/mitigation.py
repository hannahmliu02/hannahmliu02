"""
mitigation.py
-------------
Implements Option A: "Uncertainty-Triggered Measurement"

Rationale
---------
The baseline policy monitors patients when predicted risk is high, creating a
feedback loop: under-observed patients accumulate long gaps → model sees high
gap → lowers predicted risk → policy keeps measurement low.

The mitigation breaks this loop by *also* monitoring patients when:
  (a) the model is *uncertain* (entropy proxy: p*(1-p) > u0), OR
  (b) the gap since last measurement exceeds g0 steps.

Either condition forces the measurement probability to p_high regardless of
the predicted risk score.  This ensures that patients with long gaps (often
those in under-observed subgroups) still get measured, which gives the model
fresh signal and prevents risk under-estimation from self-perpetuating.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from toy.simulate_policy import run_simulation, plot_simulation


def make_uncertainty_override(
    u0: float = 0.20,
    g0: int = 3,
    p_high: float = 0.85,
) -> callable:
    """Return a measurement-probability override function.

    Signature: new_prob = fn(base_prob, pred_prob, gap) — all numpy arrays.
    """
    def override_fn(base_prob: np.ndarray, pred_prob: np.ndarray, gap: np.ndarray):
        uncertainty = pred_prob * (1.0 - pred_prob)
        trigger = (uncertainty > u0) | (gap > g0)
        return np.where(trigger, p_high, base_prob)
    return override_fn


def compare_simulations(
    model,
    df_feat: pd.DataFrame,
    sim_kwargs: dict,
    u0: float = 0.20,
    g0: int = 3,
    p_high: float = 0.85,
    out_dir: str = "outputs",
) -> dict:
    """Run baseline and mitigated simulations; produce comparison plots/table.

    Returns
    -------
    dict with scalar disparity metrics for both conditions
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # -------- Baseline -------------------------------------------------------
    df_base = run_simulation(model, df_feat, **sim_kwargs)

    # -------- Mitigated ------------------------------------------------------
    override_fn = make_uncertainty_override(u0=u0, g0=g0, p_high=p_high)
    df_mitig = run_simulation(
        model, df_feat,
        meas_prob_override_fn=override_fn,
        **sim_kwargs,
    )

    # -------- Individual trajectory plots ------------------------------------
    plot_simulation(df_base,  label="Baseline",   out_dir=out_dir, suffix="baseline")
    plot_simulation(df_mitig, label="Mitigation", out_dir=out_dir, suffix="mitigation")

    # -------- Side-by-side comparison ----------------------------------------
    panels = [
        ("meas_rate",      "Measurement Rate"),
        ("mean_pred_risk", "Mean Predicted Risk"),
        ("mean_gap",       "Mean Gap (steps)"),
    ]
    colors_g = {0: "steelblue", 1: "darkorange"}
    ls = {"Baseline": "-", "Mitigation": "--"}
    alpha = {"Baseline": 0.55, "Mitigation": 1.0}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (col, title) in zip(axes, panels):
        for cond_lbl, df_c in [("Baseline", df_base), ("Mitigation", df_mitig)]:
            for g in [0, 1]:
                sub = df_c[df_c["group"] == g].sort_values("t")
                ax.plot(
                    sub["t"], sub[col],
                    color=colors_g[g], linestyle=ls[cond_lbl],
                    linewidth=2, alpha=alpha[cond_lbl],
                    label=f"G{g} {cond_lbl}",
                )
        ax.set_title(title)
        ax.set_xlabel("Deployment Timestep")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Baseline vs Uncertainty-Triggered Mitigation  (u0={u0}, g0={g0} steps)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(f"{out_dir}/mitigation_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # -------- Summary table --------------------------------------------------
    rows = []
    for cond_lbl, df_c in [("Baseline", df_base), ("Mitigation", df_mitig)]:
        for g in [0, 1]:
            last = df_c[(df_c["group"] == g) & (df_c["t"] >= df_c["t"].max() - 4)]
            rows.append(
                {
                    "condition": cond_lbl,
                    "group": g,
                    "mean_meas_rate_last5":  last["meas_rate"].mean(),
                    "mean_pred_risk_last5":  last["mean_pred_risk"].mean(),
                    "mean_gap_last5":        last["mean_gap"].mean(),
                    "true_severity_last5":   last["true_severity"].mean(),
                }
            )
    pd.DataFrame(rows).to_csv(f"{out_dir}/mitigation_summary.csv", index=False)

    # -------- Disparity scalars ----------------------------------------------
    def _disp(df_c, col):
        # (group0 mean) - (group1 mean) over all timesteps
        g0 = df_c[df_c["group"] == 0][col].mean()
        g1 = df_c[df_c["group"] == 1][col].mean()
        return float(g0 - g1)

    summary = {
        "baseline_meas_disparity":  _disp(df_base,  "meas_rate"),
        "mitigated_meas_disparity": _disp(df_mitig, "meas_rate"),
        "baseline_risk_disparity":  _disp(df_base,  "mean_pred_risk"),
        "mitigated_risk_disparity": _disp(df_mitig, "mean_pred_risk"),
        "baseline_gap_disparity":   _disp(df_base,  "mean_gap"),
        "mitigated_gap_disparity":  _disp(df_mitig, "mean_gap"),
    }
    return summary


if __name__ == "__main__":
    from toy.generate_data import generate_data
    from toy.train_model import build_features, train_model

    df = generate_data()
    df_feat = build_features(df)
    model, metrics, _ = train_model(df_feat)

    sim_kwargs = dict(T_sim=30, p11=0.8, p01=0.2, mu0=0.0, mu1=1.0,
                      sigma=1.0, tau=0.5, p_high=0.85, p_low=0.15, seed=123)

    summary = compare_simulations(model, df_feat, sim_kwargs)
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")
