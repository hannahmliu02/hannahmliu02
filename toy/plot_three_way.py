"""
plot_three_way.py
-----------------
Three-way comparison: Baseline vs Uncertainty-Triggered vs Causal RLHF.

Usage:
    python -m toy.plot_three_way
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.lines import Line2D

from toy.generate_data import generate_data
from toy.train_model import build_features, train_model
from toy.simulate_policy import run_simulation
from toy.mitigation import make_uncertainty_override
from toy.causal_rlhf import CausalRLHFPipeline, CausalRLHFWrapper

OUT_DIR = "outputs"

PARAMS = dict(
    N=2000, T=20, p11=0.8, p01=0.2,
    meas_prob_s0=(0.30, 0.15),
    meas_prob_s1=(0.80, 0.60),
    mu0=0.0, mu1=1.0, sigma=1.0, noise_flip=0.05, seed=42,
)
SIM = dict(
    T_sim=30, p11=0.8, p01=0.2, mu0=0.0, mu1=1.0,
    sigma=1.0, tau=0.5, p_high=0.85, p_low=0.15, seed=123,
)


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # ── Data + model ──────────────────────────────────────────────────────────
    df = generate_data(**PARAMS)
    df_feat = build_features(df)
    model, _, _ = train_model(df_feat, seed=42)

    # ── Three simulations ─────────────────────────────────────────────────────
    df_base = run_simulation(model, df_feat, **SIM)

    override = make_uncertainty_override(u0=0.20, g0=3, p_high=0.85)
    df_unc = run_simulation(model, df_feat, meas_prob_override_fn=override, **SIM)

    pipeline = CausalRLHFPipeline(model)
    pipeline.train(df_feat, n_rounds=8)
    wrapper = CausalRLHFWrapper(model, pipeline.m2)
    df_rlhf = run_simulation(wrapper, df_feat, **SIM)

    # ── Plot config ───────────────────────────────────────────────────────────
    panels = [
        ("meas_rate",      "Measurement Rate"),
        ("mean_pred_risk", "Mean Predicted Risk"),
        ("mean_gap",       "Mean Gap (steps)"),
    ]
    # (label, df, linestyle, alpha)
    conditions = [
        ("Baseline (M1)",         df_base, "-",  0.55),
        ("Uncertainty-Triggered", df_unc,  "--", 0.85),
        ("Causal RLHF (M2)",      df_rlhf, ":",  1.0),
    ]
    colors = {0: "steelblue", 1: "darkorange"}

    # ── Combined 2×3 figure ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    axes_top, axes_bot = axes[0], axes[1]

    # Metrics where Causal RLHF is identical to Uncertainty-Triggered
    rlhf_identical_cols = {"meas_rate", "mean_gap"}

    # Row 1 — line charts
    for ax, (col, title) in zip(axes_top, panels):
        for cond_lbl, df_c, ls, alpha in conditions:
            if cond_lbl == "Causal RLHF (M2)" and col in rlhf_identical_cols:
                continue
            for g in [0, 1]:
                sub = df_c[df_c["group"] == g].sort_values("t")
                ax.plot(
                    sub["t"], sub[col],
                    color=colors[g], linestyle=ls, linewidth=2, alpha=alpha,
                    label=f"G{g} {cond_lbl}",
                )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Deployment Timestep")
        ax.grid(True, alpha=0.3)
        if col in rlhf_identical_cols:
            ax.text(0.98, 0.04,
                    "Causal RLHF = Uncertainty-Triggered\n(measurement policy unchanged)",
                    transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
                    color="dimgray", style="italic",
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray",
                              alpha=0.8))

    legend_elements = [
        Line2D([0], [0], color="steelblue",  lw=2, label="Group 0"),
        Line2D([0], [0], color="darkorange", lw=2, label="Group 1 (under-obs)"),
        Line2D([0], [0], color="gray", lw=2, ls="-",  alpha=0.55, label="Baseline (M1)"),
        Line2D([0], [0], color="gray", lw=2, ls="--", alpha=0.85, label="Uncertainty-Triggered"),
        Line2D([0], [0], color="gray", lw=2, ls=":",  alpha=1.0,  label="Causal RLHF (M2)"),
    ]
    axes_top[2].legend(handles=legend_elements, fontsize=8, loc="upper right")

    # Row 2 — disparity bar charts
    metric_keys   = ["meas_rate",  "mean_pred_risk", "mean_gap"]
    metric_labels = [
        "Measurement Rate\nDisparity (G0−G1)",
        "Predicted Risk\nDisparity (G0−G1)",
        "Gap Length\nDisparity (G0−G1)",
    ]
    cond_labels = ["Baseline\n(M1)", "Uncertainty-\nTriggered", "Causal\nRLHF (M2)"]
    bar_colors  = ["#d9534f", "#5bc0de", "#5cb85c"]

    for ax, col, mlabel in zip(axes_bot, metric_keys, metric_labels):
        disps = []
        for _, df_c, *_ in conditions:
            g0 = df_c[df_c["group"] == 0][col].mean()
            g1 = df_c[df_c["group"] == 1][col].mean()
            disps.append(g0 - g1)
        bars = ax.bar(cond_labels, disps, color=bar_colors, edgecolor="white", width=0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(mlabel, fontsize=11)
        ax.set_ylabel("G0 − G1 (avg over simulation)")
        for bar, val in zip(bars, disps):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + (0.001 if val >= 0 else -0.003),
                f"{val:+.4f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=9, fontweight="bold",
            )
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(
        "Three-Way Comparison: Baseline vs Uncertainty-Triggered vs Causal RLHF\n"
        "Solid=Baseline | Dashed=Uncertainty-Triggered | Dotted=Causal RLHF",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/three_way_combined.png", dpi=130, bbox_inches="tight")
    print(f"Saved {OUT_DIR}/three_way_combined.png")

    # Also save the individual plots
    fig_top, axes_t = plt.subplots(1, 3, figsize=(17, 5))
    for ax_new, ax_src in zip(axes_t, axes_top):
        for line in ax_src.get_lines():
            ax_new.plot(line.get_xdata(), line.get_ydata(),
                        color=line.get_color(), linestyle=line.get_linestyle(),
                        linewidth=line.get_linewidth(), alpha=line.get_alpha())
        ax_new.set_title(ax_src.get_title(), fontsize=12)
        ax_new.set_xlabel(ax_src.get_xlabel())
        ax_new.grid(True, alpha=0.3)
    axes_t[2].legend(handles=legend_elements, fontsize=8, loc="upper right")
    fig_top.suptitle(
        "Three-Way Comparison: Baseline vs Uncertainty-Triggered vs Causal RLHF\n"
        "Solid=Baseline | Dashed=Uncertainty-Triggered | Dotted=Causal RLHF",
        fontsize=12,
    )
    fig_top.tight_layout()
    fig_top.savefig(f"{OUT_DIR}/three_way_comparison.png", dpi=130, bbox_inches="tight")
    print(f"Saved {OUT_DIR}/three_way_comparison.png")

    fig_bot, axes_b = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col, mlabel in zip(axes_b, metric_keys, metric_labels):
        disps = []
        for _, df_c, *_ in conditions:
            g0 = df_c[df_c["group"] == 0][col].mean()
            g1 = df_c[df_c["group"] == 1][col].mean()
            disps.append(g0 - g1)
        bars = ax.bar(cond_labels, disps, color=bar_colors, edgecolor="white", width=0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(mlabel, fontsize=11)
        ax.set_ylabel("G0 − G1 (avg over simulation)")
        for bar, val in zip(bars, disps):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + (0.001 if val >= 0 else -0.003),
                f"{val:+.4f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=9, fontweight="bold",
            )
        ax.grid(True, alpha=0.2, axis="y")
    fig_bot.suptitle(
        "Disparity Summary: All Three Conditions\n"
        "(positive = Group 1 under-served; for Gap, negative = Group 1 has longer gap)",
        fontsize=11,
    )
    fig_bot.tight_layout()
    fig_bot.savefig(f"{OUT_DIR}/disparity_bar_chart.png", dpi=130, bbox_inches="tight")
    print(f"Saved {OUT_DIR}/disparity_bar_chart.png")


if __name__ == "__main__":
    main()
