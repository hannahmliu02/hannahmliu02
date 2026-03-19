"""
audit.py
--------
Observability audit via counterfactual masking.

For every test-set record we ask: "How much does the predicted risk change
if we *pretend* this timestep was unobserved?"

Concretely:
  original features  -> risk_orig  = model.predict_proba(features)[:,1]
  masked features    -> risk_mask  = model.predict_proba(features_masked)[:,1]
  delta_risk         = risk_orig - risk_mask

Masking means:
  - missing_now  <- 1
  - gap          <- gap + 1   (one extra step without measurement)
  - x_ffill stays the same (last previously observed value, unchanged)
  - meas_count stays the same

A positive delta_risk means the model assigns higher risk when the patient
IS observed (compared to when they are not).  If delta_risk is systematically
higher for one subgroup, the model is more sensitive to observability for that
group — which can amplify under-observation.
"""

import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from toy.train_model import FEATURE_COLS


def compute_delta_risk(model, df_feat: pd.DataFrame) -> pd.DataFrame:
    """Return df_feat with additional columns: risk_orig, risk_mask, delta_risk."""
    X_orig = df_feat[FEATURE_COLS].values.copy()

    # Build masked feature matrix
    X_mask = X_orig.copy()
    col_idx = {c: i for i, c in enumerate(FEATURE_COLS)}

    # For rows currently observed (missing_now == 0), flip to missing
    # and increment gap by 1.  For already-missing rows, gap += 1.
    X_mask[:, col_idx["missing_now"]] = 1
    X_mask[:, col_idx["gap"]] = X_orig[:, col_idx["gap"]] + 1

    risk_orig = model.predict_proba(X_orig)[:, 1]
    risk_mask = model.predict_proba(X_mask)[:, 1]
    delta_risk = risk_orig - risk_mask

    out = df_feat.copy()
    out["risk_orig"] = risk_orig
    out["risk_mask"] = risk_mask
    out["delta_risk"] = delta_risk
    return out


def run_audit(
    model,
    df_feat: pd.DataFrame,
    out_dir: str = "outputs",
) -> dict:
    """Run full audit, produce plots, return summary statistics."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    df_audit = compute_delta_risk(model, df_feat)

    # ------------------------------------------------------------------ #
    # 1. Histogram of delta_risk overall and by subgroup
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)

    for ax, (label, subset) in zip(
        axes,
        [
            ("Overall", df_audit),
            ("Group 0", df_audit[df_audit["group"] == 0]),
            ("Group 1 (under-observed)", df_audit[df_audit["group"] == 1]),
        ],
    ):
        ax.hist(subset["delta_risk"], bins=40, color="steelblue", edgecolor="white", alpha=0.8)
        mean_val = subset["delta_risk"].mean()
        ax.axvline(mean_val, color="crimson", linestyle="--", linewidth=1.5,
                   label=f"mean={mean_val:.3f}")
        ax.set_title(label)
        ax.set_xlabel("delta_risk  (risk_obs − risk_unobs)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    fig.suptitle("Counterfactual Masking Audit: Δrisk = risk(observed) − risk(masked)", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/audit_delta_risk_hist.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 2. delta_risk vs gap length (binned mean ± std)
    # ------------------------------------------------------------------ #
    df_audit["gap_bin"] = pd.cut(df_audit["gap"], bins=[0, 1, 2, 3, 5, 10, 20],
                                  right=True, labels=["0-1","1-2","2-3","3-5","5-10","10+"])
    gap_stats = (
        df_audit.groupby(["gap_bin", "group"], observed=False)["delta_risk"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {0: "steelblue", 1: "darkorange"}
    labels_g = {0: "Group 0", 1: "Group 1 (under-obs)"}
    for g in [0, 1]:
        sub = gap_stats[gap_stats["group"] == g]
        xs = range(len(sub))
        ax.errorbar(
            xs, sub["mean"], yerr=sub["std"],
            fmt="-o", color=colors[g], label=labels_g[g], capsize=4
        )
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub["gap_bin"].astype(str), rotation=30)
    ax.set_xlabel("Gap length (steps since last measurement)")
    ax.set_ylabel("Mean Δrisk")
    ax.set_title("Δrisk vs Gap Length by Subgroup")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/audit_delta_risk_vs_gap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 3. Summary statistics
    # ------------------------------------------------------------------ #
    summary_rows = []
    for g in [0, 1]:
        sub = df_audit[df_audit["group"] == g]["delta_risk"]
        summary_rows.append(
            {
                "group": g,
                "mean_delta_risk": sub.mean(),
                "median_delta_risk": sub.median(),
                "std_delta_risk": sub.std(),
                "pct_positive": (sub > 0).mean(),
                "n": len(sub),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{out_dir}/audit_summary.csv", index=False)

    # Also return as dict for console printing
    summary = {
        "group0_mean_delta": float(df_audit[df_audit["group"] == 0]["delta_risk"].mean()),
        "group1_mean_delta": float(df_audit[df_audit["group"] == 1]["delta_risk"].mean()),
    }
    return summary


def run_multistep_masking(
    model,
    df_feat: pd.DataFrame,
    K_max: int = 10,
    highlight_k: tuple = (3, 5, 10),
    out_dir: str = "outputs",
) -> pd.DataFrame:
    """Multi-step masking: simulate k consecutive unmeasured steps and track
    how predicted risk evolves, comparing Group 0 vs Group 1.

    Two experimental conditions are evaluated:

    1. **Actual-state start** — each patient initialised from their real
       last-timestep features (x_ffill, gap, meas_count as accumulated during
       training).  Group 1 already has higher gaps at k=0 because they were
       systematically under-measured.

    2. **Controlled start** — gap and missing_now are reset to 0 / "just
       measured" for all patients, while x_ffill and meas_count are kept.
       This isolates the effect of stale clinical readings from the pure
       measurement-gap bias.  In this condition the model is group-blind
       through features (no group input), so any residual gap comes only
       from differing x_ffill distributions caused by historical under-
       measurement.

    For each condition we record, per (k, group):
      mean_risk, std_risk, mean_delta_risk (= risk(k) − risk(0))

    Produces:
      outputs/audit_multistep_masking.png   — 3-panel actual-state figure
      outputs/audit_multistep_controlled.png — 2-panel controlled figure
      outputs/audit_multistep_summary.csv   — per-(k, group, condition) table
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    col_idx = {c: i for i, c in enumerate(FEATURE_COLS)}

    # Use the LAST timestep per patient (one row per patient) so that
    # accumulated historical gaps are fully reflected.
    last_step = (
        df_feat.sort_values("t")
        .groupby("patient_id", sort=True)
        .last()
        .reset_index()
    )

    groups = last_step["group"].values.astype(int)
    X_actual = last_step[FEATURE_COLS].values.astype(float)

    # Controlled: reset gap=0, missing_now=0 (fresh measurement for everyone)
    X_controlled = X_actual.copy()
    X_controlled[:, col_idx["gap"]] = 0
    X_controlled[:, col_idx["missing_now"]] = 0

    def _simulate_steps(X_base, K_max, groups):
        """For each patient, compute predicted risk at k=0..K_max extra
        unmeasured steps. Returns a list of dicts."""
        risk_0 = model.predict_proba(X_base)[:, 1]
        records = []
        for k in range(K_max + 1):
            X_k = X_base.copy()
            # After k>0 extra unmeasured steps: missing_now=1, gap increases k
            if k > 0:
                X_k[:, col_idx["missing_now"]] = 1
            X_k[:, col_idx["gap"]] = X_base[:, col_idx["gap"]] + k
            risk_k = model.predict_proba(X_k)[:, 1]
            for i in range(len(groups)):
                records.append({
                    "k":           k,
                    "group":       int(groups[i]),
                    "risk":        float(risk_k[i]),
                    "delta_risk":  float(risk_k[i] - risk_0[i]),
                })
        return records

    records_actual = _simulate_steps(X_actual,     K_max, groups)
    records_ctrl   = _simulate_steps(X_controlled, K_max, groups)

    def _agg(records):
        df = pd.DataFrame(records)
        return (
            df.groupby(["k", "group"])
            .agg(
                mean_risk=("risk",       "mean"),
                std_risk=("risk",        "std"),
                mean_delta_risk=("delta_risk", "mean"),
                std_delta_risk=("delta_risk",  "std"),
                n=("risk", "count"),
            )
            .reset_index()
        )

    agg_actual = _agg(records_actual)
    agg_ctrl   = _agg(records_ctrl)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    COLORS = {0: "steelblue", 1: "darkorange"}
    GLABELS = {0: "Group 0", 1: "Group 1 (under-obs)"}

    def _shade(ax, sub, col, n, color, alpha=0.15):
        se = sub[f"std_{col}"] / np.sqrt(n)
        ax.fill_between(sub["k"],
                        sub[f"mean_{col}"] - 2 * se,
                        sub[f"mean_{col}"] + 2 * se,
                        alpha=alpha, color=color)

    def _vlines(ax, highlight_k, K_max):
        for kv in highlight_k:
            if kv <= K_max:
                ax.axvline(kv, color="gray", linewidth=0.9,
                           linestyle="--", alpha=0.55)

    # ================================================================== #
    # Figure 1 — Actual-state start (3 panels)
    # ================================================================== #
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for g in [0, 1]:
        sub = agg_actual[agg_actual["group"] == g].sort_values("k")
        n_g = int(sub["n"].iloc[0])
        c   = COLORS[g]

        # Panel 1: Absolute risk
        axes[0].plot(sub["k"], sub["mean_risk"],
                     color=c, linewidth=2.5, marker="o", markersize=4,
                     label=GLABELS[g])
        _shade(axes[0], sub, "risk", n_g, c)

        # Panel 2: Risk degradation Δrisk(k) = risk(k) − risk(0)
        axes[1].plot(sub["k"], sub["mean_delta_risk"],
                     color=c, linewidth=2.5, marker="o", markersize=4,
                     label=GLABELS[g])
        _shade(axes[1], sub, "delta_risk", n_g, c)

    # Panel 3: Inter-group risk gap  G0 − G1
    g0_risk = agg_actual[agg_actual["group"] == 0].sort_values("k")["mean_risk"].values
    g1_risk = agg_actual[agg_actual["group"] == 1].sort_values("k")["mean_risk"].values
    ks = agg_actual[agg_actual["group"] == 0].sort_values("k")["k"].values
    axes[2].plot(ks, g0_risk - g1_risk,
                 color="crimson", linewidth=2.5, marker="o", markersize=4,
                 label="Gap (G0 − G1)")
    axes[2].axhline(0, color="black", linewidth=0.8, linestyle=":")
    # Annotate at highlight_k
    for kv in highlight_k:
        if kv <= K_max:
            idx = np.where(ks == kv)[0]
            if len(idx):
                gap_val = (g0_risk - g1_risk)[idx[0]]
                axes[2].annotate(f"k={kv}\n{gap_val:+.3f}",
                                 xy=(kv, gap_val),
                                 xytext=(kv + 0.4, gap_val + 0.002),
                                 fontsize=7.5, color="crimson")

    for ax in axes:
        _vlines(ax, highlight_k, K_max)
        ax.set_xlabel("Extra consecutive unmeasured steps  k")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_title("Mean Predicted Risk\n(actual historical starting state)")
    axes[0].set_ylabel("Mean predicted risk")

    axes[1].set_title("Risk Degradation  Δrisk(k) = risk(k) − risk(0)\n"
                      "(relative drop from each patient's own baseline)")
    axes[1].set_ylabel("Δ risk")
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle=":")

    axes[2].set_title("Inter-Group Risk Gap  (G0 − G1)\n"
                      "vs. consecutive unmeasured steps")
    axes[2].set_ylabel("Risk gap (G0 − G1)")

    fig.suptitle(
        "Multi-Step Masking Audit — Actual Historical Starting State\n"
        "Dashed verticals mark k = 3, 5, 10",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(f"{out_dir}/audit_multistep_masking.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ================================================================== #
    # Figure 2 — Controlled start (gap reset to 0, 2 panels)
    # ================================================================== #
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5))

    for g in [0, 1]:
        sub = agg_ctrl[agg_ctrl["group"] == g].sort_values("k")
        n_g = int(sub["n"].iloc[0])
        c   = COLORS[g]

        axes2[0].plot(sub["k"], sub["mean_risk"],
                      color=c, linewidth=2.5, marker="o", markersize=4,
                      label=GLABELS[g])
        _shade(axes2[0], sub, "risk", n_g, c)

        axes2[1].plot(sub["k"], sub["mean_delta_risk"],
                      color=c, linewidth=2.5, marker="o", markersize=4,
                      label=GLABELS[g])
        _shade(axes2[1], sub, "delta_risk", n_g, c)

    for ax in axes2:
        _vlines(ax, highlight_k, K_max)
        ax.set_xlabel("Extra consecutive unmeasured steps  k")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes2[0].set_title("Mean Predicted Risk\n(controlled: gap reset to 0 for all)")
    axes2[0].set_ylabel("Mean predicted risk")

    axes2[1].set_title("Risk Degradation  Δrisk(k)\n(controlled start)")
    axes2[1].set_ylabel("Δ risk")
    axes2[1].axhline(0, color="black", linewidth=0.8, linestyle=":")

    fig2.suptitle(
        "Multi-Step Masking — Controlled Start (gap=0 for all patients)\n"
        "Any residual inter-group gap here reflects stale x_ffill, not gap history",
        fontsize=11,
    )
    fig2.tight_layout()
    fig2.savefig(f"{out_dir}/audit_multistep_controlled.png",
                 dpi=120, bbox_inches="tight")
    plt.close(fig2)

    # ================================================================== #
    # CSV summary
    # ================================================================== #
    agg_actual["condition"] = "actual"
    agg_ctrl["condition"]   = "controlled"
    combined = pd.concat([agg_actual, agg_ctrl], ignore_index=True)
    combined.to_csv(f"{out_dir}/audit_multistep_summary.csv", index=False)

    # Return actual-state aggregate for console printing
    return agg_actual


if __name__ == "__main__":
    from toy.generate_data import generate_data
    from toy.train_model import build_features, train_model

    df = generate_data()
    df_feat = build_features(df)
    model, metrics, df_test = train_model(df_feat)
    summary = run_audit(model, df_test)
    print("Audit summary:", summary)
    ms = run_multistep_masking(model, df_feat)
    print(ms[ms["k"].isin([0, 3, 5, 10])])
