"""
simulate_policy.py
------------------
Score-based monitoring policy simulation.

Key design: the simulation initialises from the *actual* end-of-training
patient states so that the measurement-history bias is already baked in
at time t=0.  Group-1 patients arrive in the deployment period with longer
historical gaps and more missing indicators (because they were
systematically under-measured during data collection).  The trained model
has learnt "long gap / missing → lower risk", so it immediately assigns
lower predicted risk to group-1, which causes the risk-driven policy to
measure them less, which widens the gap further — the feedback loop.

Deployment loop (T_sim steps):
  1. Build features from current patient state.
  2. Predict risk = model.predict_proba(features)[:,1].
  3. Policy: if risk > tau  →  measurement prob = p_high
             else           →  measurement prob = p_low
     (optionally overridden by a mitigation function)
  4. Draw M_t ~ Bernoulli(measurement_prob).
  5. If M_t=1: observe X_t ~ N(mu1|S=1, mu0|S=0).
  6. Update running features (x_ffill, gap, meas_count, missing_now).
  7. Markov-transition severity state S.

Returns per-(t, group) aggregate statistics for plotting.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


SIM_DEFAULTS = dict(
    T_sim=30,
    p11=0.8,
    p01=0.2,
    mu0=0.0,
    mu1=1.0,
    sigma=1.0,
    tau=0.5,
    p_high=0.85,
    p_low=0.15,
    seed=123,   # separate from data-gen seed for independence
)


def run_simulation(
    model,
    df_feat: pd.DataFrame,       # feature dataframe from build_features()
    T_sim: int = 30,
    p11: float = 0.8,
    p01: float = 0.2,
    mu0: float = 0.0,
    mu1: float = 1.0,
    sigma: float = 1.0,
    tau: float = 0.5,
    p_high: float = 0.85,
    p_low: float = 0.15,
    seed: int = 123,
    meas_prob_override_fn=None,
) -> pd.DataFrame:
    """Run score-based monitoring simulation initialised from training states.

    Parameters
    ----------
    model : fitted sklearn estimator (predict_proba)
    df_feat : feature dataframe produced by build_features(); the simulation
              initialises each patient's state from their LAST timestep in
              this dataframe (x_ffill, gap, meas_count, missing_now, S).
    T_sim : number of deployment timesteps to simulate
    meas_prob_override_fn : optional callable(base_prob, pred_prob, gap)
              used by mitigation strategies

    Returns
    -------
    pd.DataFrame with columns:
        t, group, meas_rate, mean_pred_risk, mean_gap, true_severity
    (one row per (t, group) pair, t ∈ [0, T_sim))
    """
    rng = np.random.default_rng(seed)

    # -------- Initialise from last timestep of each patient ---------------
    last_step = (
        df_feat.sort_values("t")
        .groupby("patient_id", sort=True)
        .last()
        .reset_index()
    )
    last_step = last_step.sort_values("patient_id").reset_index(drop=True)

    N = len(last_step)
    groups     = last_step["group"].values.astype(int)
    S          = last_step["S"].values.astype(int)
    x_ffill    = last_step["x_ffill"].values.astype(float)
    gap        = last_step["gap"].values.astype(int)
    meas_count = last_step["meas_count"].values.astype(int)
    missing_now = last_step["missing_now"].values.astype(int)

    time_records = []

    for t in range(T_sim):
        # -------- 1. Build features [x_ffill, missing_now, gap, meas_count] --
        features = np.column_stack([x_ffill, missing_now, gap, meas_count]).astype(float)

        # -------- 2. Predict risk -------------------------------------------
        pred_prob = model.predict_proba(features)[:, 1]

        # -------- 3. Measurement probability (policy) ----------------------
        base_meas_prob = np.where(pred_prob > tau, p_high, p_low)

        if meas_prob_override_fn is not None:
            meas_prob = meas_prob_override_fn(base_meas_prob, pred_prob, gap)
        else:
            meas_prob = base_meas_prob

        # -------- 4. Draw measurements -------------------------------------
        M = (rng.random(N) < meas_prob).astype(int)
        mu_vec = np.where(S == 1, mu1, mu0)
        X_obs = rng.normal(mu_vec, sigma)

        measured = M == 1
        x_ffill[measured]    = X_obs[measured]
        gap[measured]        = 0
        meas_count[measured] += 1
        missing_now[measured] = 0

        gap[~measured]        += 1
        missing_now[~measured] = 1

        # -------- 5. Record per-group stats --------------------------------
        for g in [0, 1]:
            mask = groups == g
            time_records.append(
                {
                    "t": t,
                    "group": g,
                    "meas_rate":      float(M[mask].mean()),
                    "mean_pred_risk": float(pred_prob[mask].mean()),
                    "mean_gap":       float(gap[mask].mean()),
                    "true_severity":  float(S[mask].mean()),
                }
            )

        # -------- 6. Severity transition -----------------------------------
        trans_prob = np.where(S == 1, p11, p01)
        S = (rng.random(N) < trans_prob).astype(int)

    return pd.DataFrame(time_records)


def plot_simulation(
    df_traj: pd.DataFrame,
    label: str = "Baseline",
    out_dir: str = "outputs",
    suffix: str = "baseline",
):
    """Save four-panel trajectory plot for one simulation condition."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    panels = [
        ("meas_rate",      "Measurement Rate"),
        ("mean_pred_risk", "Mean Predicted Risk"),
        ("mean_gap",       "Mean Gap (steps since last obs.)"),
        ("true_severity",  "True Severity Prevalence"),
    ]
    colors = {0: "steelblue", 1: "darkorange"}
    linestyles = {0: "-", 1: "--"}
    group_labels = {0: "Group 0", 1: "Group 1 (under-obs)"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (col, title) in zip(axes, panels):
        for g in [0, 1]:
            sub = df_traj[df_traj["group"] == g].sort_values("t")
            ax.plot(
                sub["t"], sub[col],
                color=colors[g], linestyle=linestyles[g],
                linewidth=2, marker="o", markersize=4,
                label=group_labels[g],
            )
        ax.set_title(title)
        ax.set_xlabel("Deployment Timestep")
        ax.set_ylabel(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Score-Based Monitoring Policy — {label}", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/sim_{suffix}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    from toy.generate_data import generate_data
    from toy.train_model import build_features, train_model

    df = generate_data()
    df_feat = build_features(df)
    model, metrics, _ = train_model(df_feat)

    df_traj = run_simulation(model, df_feat)
    plot_simulation(df_traj, label="Baseline", suffix="baseline")
    print(df_traj.tail(10))
