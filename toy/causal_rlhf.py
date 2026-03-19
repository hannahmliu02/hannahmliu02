"""
causal_rlhf.py
--------------
Causal RLHF pipeline as a bias mitigation technique.

Graph structure (three sequential causal chains):

  t=1:  x  --> M1 --> y1          (M1 observed)
  t=2:  M2 --> y2 --> RM1         (y2 observed)
        h  --> M1 --> y1          (M1 observed)
  t=3:  y1 --> RM2 --> M2         (RM2 observed)

Nodes:
  x   : patient feature vectors (human input)
  h   : confounding variables (group membership + measurement bias patterns)
  M1  : primary risk model (frozen logistic regression, always observed)
  M2  : debiasing model (updated each round via RM2's training signal)
  y1  : M1's predictions (biased by h via the h --> M1 --> y1 path)
  y2  : M2's predictions (debiased; M2 takes y1 + proxies as input)
  RM1 : fairness auditor — evaluates y2 given h, measures residual group disparity
  RM2 : debiasing trainer — diagnoses h-caused bias in y1, trains M2 each round

Training loop (R rounds):
  For each round r:
    [t=1] x  --> M1 --> y1 :  apply frozen M1, record biased predictions (M1 observed)
    [t=3] y1 --> RM2 --> M2:  RM2 diagnoses residual bias in y2_{r-1} relative to Y,
                               produces correction targets, updates M2 (RM2 observed)
    [t=2] M2 --> y2 --> RM1:  M2 produces debiased y2, RM1 scores fairness (y2 observed)
          h  --> M1 --> y1 :  track h's continued influence on M1 (M1 observed)

Key design choices:
  - M1 is FROZEN throughout (like a pre-trained base model in RLHF).
  - M2 takes (y1, gap, missing_now) as features, NOT group directly, so it
    learns to debias from observable proxies of h rather than from h itself.
  - RM2 uses group labels (available at training time) to identify h-caused
    bias and produce correction targets via linear residual regression.
  - Each round RM2 evaluates the CURRENT M2 output y2_{r-1}, not y1, enabling
    genuine iterative refinement of residual bias over rounds.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from toy.train_model import FEATURE_COLS
from toy.simulate_policy import run_simulation, plot_simulation


# ============================================================================
# RM1 — Fairness Auditor
# ============================================================================

class RewardModel1:
    """RM1: Fairness auditor.

    Evaluates y2 (M2's predictions) for residual bias from confounders h.
    Observability: y2 is the observed middle node of  M2 --> y2 --> RM1.

    Computes:
      - raw_disparity          = mean(y2 | g=0) - mean(y2 | g=1)
      - severity_adj_disparity = residual disparity after removing true
                                 severity differences between groups
      - rm1_score              = 1 - |severity_adj_disparity|  (higher = fairer)

    Also computes the same metrics on y1 (M1's biased predictions) so the
    caller can compare fairness before and after debiasing.
    """

    def evaluate(
        self,
        y1: np.ndarray,
        y2: np.ndarray,
        groups: np.ndarray,
        severity: np.ndarray,
    ) -> dict:
        """Score fairness of y2 (and y1 as baseline) given confounders h=groups."""

        def _disparity(preds):
            raw = float(preds[groups == 0].mean() - preds[groups == 1].mean())
            # severity-adjusted: remove the part explained by true severity gap
            res0 = preds[groups == 0] - severity[groups == 0]
            res1 = preds[groups == 1] - severity[groups == 1]
            adj = float(res0.mean() - res1.mean())
            return raw, adj

        y1_raw, y1_adj = _disparity(y1)
        y2_raw, y2_adj = _disparity(y2)

        return {
            "y1_raw_disparity":  y1_raw,
            "y1_adj_disparity":  y1_adj,
            "y2_raw_disparity":  y2_raw,
            "y2_adj_disparity":  y2_adj,
            "rm1_score":         float(1.0 - abs(y2_adj)),
            "disparity_reduction": float(abs(y1_adj) - abs(y2_adj)),
        }


# ============================================================================
# RM2 — Debiasing Trainer
# ============================================================================

class RewardModel2:
    """RM2: Debiasing trainer.

    Evaluates y1 (M1's biased predictions) for h-caused bias and produces
    correction targets to train M2.

    Observability: RM2 is the observed middle node of  y1 --> RM2 --> M2.

    Mechanism — counterfactual gap correction:
      1. For every patient compute y1_fair = M1(X with gap=0, missing_now=0).
         This is the counterfactual prediction M1 would make if the patient had
         just been measured (no gap bias). RM2 accesses M1 directly for this.
      2. correction[i] = alpha * (y1_fair[i] - y_current[i])
         For high-gap patients (disproportionately in group 1), y1_fair > y1
         because gap and missing_now carry negative coefficients in M1, so the
         correction is upward, undoing gap-induced under-estimation.
      3. y2_target = clip(y_current + correction, 0, 1)
         On round 0:  y_current = y1.
         On later rounds: y_current = y2_{r-1}, so RM2 adds only the residual
         correction still needed, enabling genuine iterative refinement.

    alpha follows an annealing schedule (0.4 → 1.0 over 5 rounds) so M2
    converges gradually toward fully gap-corrected predictions.
    """

    def __init__(self, m1_model, feature_cols):
        self.m1 = m1_model
        self.feature_cols = feature_cols
        self.last_r2 = None
        self._round = 0

    def compute_correction(
        self,
        y_current: np.ndarray,
        X: np.ndarray,
        groups: np.ndarray,
        gap: np.ndarray,
        missing_now: np.ndarray,
    ) -> tuple:
        """Compute counterfactual gap correction.

        Parameters
        ----------
        y_current   : current predictions (y1 round 0, y2_{r-1} later)
        X           : full feature matrix [x_ffill, missing_now, gap, meas_count]
        groups      : confounder h (used for R² diagnostic only)
        gap         : steps since last measurement
        missing_now : current missingness indicator

        Returns
        -------
        correction : per-sample additive correction
        y2_target  : M2 training targets = clip(y_current + correction, 0, 1)
        """
        col_missing = self.feature_cols.index("missing_now")
        col_gap     = self.feature_cols.index("gap")

        # Counterfactual: what would M1 predict with gap=0, missing_now=0?
        X_fair = X.copy().astype(float)
        X_fair[:, col_missing] = 0
        X_fair[:, col_gap]     = 0
        y1_fair = self.m1.predict_proba(X_fair)[:, 1]

        # Alpha annealing: start at 0.4, increase by 0.12 each round, cap at 1.0
        alpha = min(0.4 + self._round * 0.12, 1.0)
        self._round += 1

        correction = alpha * (y1_fair - y_current)

        # R² diagnostic: fraction of correction variance explained by group (h)
        try:
            lm = LinearRegression().fit(groups.reshape(-1, 1), correction)
            ss_res = np.sum((correction - lm.predict(groups.reshape(-1, 1))) ** 2)
            ss_tot = np.sum((correction - correction.mean()) ** 2)
            self.last_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        except Exception:
            self.last_r2 = 0.0

        y2_target = np.clip(y_current + correction, 0.0, 1.0)
        return correction, y2_target


# ============================================================================
# M2 — Debiasing Model
# ============================================================================

class DebiasingModel:
    """M2: maps M1's biased predictions to fairer predictions.

    Input features: (y1, gap, missing_now, y1×gap, y1×missing_now)
    Output: y2  (debiased predicted risk, in [0,1])

    Trained by RM2 to predict y2_target = y1 + correction.
    M2 intentionally does NOT receive group as a feature — it learns
    to debias from observable proxies (gap, missing_now) that carry
    the h-mediated measurement bias signal.

    To enable iterative refinement over multiple rounds, M2 blends
    the new fitted model with its current predictions using a learning
    rate `lr` (default 0.6), so improvements accumulate gradually.
    """

    def __init__(self):
        self._model = None
        self._scaler = StandardScaler()
        self.is_fitted = False

    def _features(self, y1, gap, missing_now):
        return np.column_stack([
            y1,
            gap,
            missing_now,
            y1 * gap,
            y1 * missing_now,
        ]).astype(float)

    def fit(
        self,
        y1: np.ndarray,
        gap: np.ndarray,
        missing_now: np.ndarray,
        y2_target: np.ndarray,
        lr: float = 0.6,
    ):
        """Train M2, blending with prior state for gradual convergence."""
        X = self._features(y1, gap, missing_now)
        X_scaled = self._scaler.fit_transform(X)

        new_model = LinearRegression()
        new_model.fit(X_scaled, y2_target)

        if self.is_fitted:
            # Soft update: blend new target with current M2 output
            y2_current = np.clip(self._model.predict(X_scaled), 0.0, 1.0)
            y2_new = np.clip(new_model.predict(X_scaled), 0.0, 1.0)
            blended = (1.0 - lr) * y2_current + lr * y2_new

            blend_model = LinearRegression()
            blend_model.fit(X_scaled, blended)
            self._model = blend_model
        else:
            self._model = new_model

        self.is_fitted = True

    def predict(self, y1: np.ndarray, gap: np.ndarray, missing_now: np.ndarray) -> np.ndarray:
        """Apply M2 debiasing: y1 --> y2."""
        if not self.is_fitted:
            return y1.copy()
        X = self._features(y1, gap, missing_now)
        X_scaled = self._scaler.transform(X)
        return np.clip(self._model.predict(X_scaled), 0.0, 1.0)


# ============================================================================
# Causal RLHF Pipeline
# ============================================================================

class CausalRLHFPipeline:
    """Three-timestep causal RLHF bias-mitigation pipeline.

    Causal chains:
      t=1:  x  --> M1 --> y1   (M1 observed)
      t=2:  M2 --> y2 --> RM1  (y2 observed)
            h  --> M1 --> y1   (M1 observed)
      t=3:  y1 --> RM2 --> M2  (RM2 observed)

    M1 is always observed (appears in both t=1 and t=2 chains).
    """

    def __init__(self, m1_model, feature_cols=None):
        self.m1 = m1_model                                        # frozen primary model
        self.m2 = DebiasingModel()                                # debiasing model updated each round
        self.rm1 = RewardModel1()                                 # fairness auditor
        self.feature_cols = feature_cols or FEATURE_COLS
        self.rm2 = RewardModel2(m1_model, self.feature_cols)      # debiasing trainer (needs M1)
        self.history = []                                         # per-round metrics

    # ── t=1: x --> M1 --> y1 (M1 observed) ──────────────────────────────────
    def _t1_m1_predict(self, X: np.ndarray) -> np.ndarray:
        """Apply frozen M1 to produce biased predictions y1. M1 is observed."""
        return self.m1.predict_proba(X)[:, 1]

    # ── t=3: y1 --> RM2 --> M2 (RM2 observed) ───────────────────────────────
    def _t3_rm2_update_m2(
        self,
        y_current: np.ndarray,
        y1: np.ndarray,
        X: np.ndarray,
        groups: np.ndarray,
        gap: np.ndarray,
        missing_now: np.ndarray,
    ) -> np.ndarray:
        """RM2 diagnoses bias in y_current, updates M2. RM2 is observed."""
        correction, y2_target = self.rm2.compute_correction(
            y_current, X, groups, gap, missing_now
        )
        self.m2.fit(y1, gap, missing_now, y2_target)
        return y2_target

    # ── t=2: M2 --> y2 --> RM1  and  h --> M1 --> y1 (y2 and M1 observed) ──
    def _t2_m2_predict_and_audit(
        self,
        X: np.ndarray,
        groups: np.ndarray,
        gap: np.ndarray,
        missing_now: np.ndarray,
        severity: np.ndarray,
    ) -> tuple:
        """M2 produces y2; RM1 evaluates fairness given h=groups."""
        y1 = self._t1_m1_predict(X)   # h --> M1 --> y1 (M1 observed)
        y2 = self.m2.predict(y1, gap, missing_now)
        rm1_scores = self.rm1.evaluate(y1, y2, groups, severity)
        return y1, y2, rm1_scores

    # ── Full round ────────────────────────────────────────────────────────────
    def _run_round(self, df_feat: pd.DataFrame, round_idx: int) -> dict:
        """Execute one full round of the three-timestep causal RLHF pipeline."""
        X = df_feat[self.feature_cols].values
        groups = df_feat["group"].values.astype(int)
        gap = df_feat["gap"].values.astype(int)
        missing_now = df_feat["missing_now"].values.astype(int)
        S = df_feat["S"].values.astype(int)

        # t=1: x --> M1 --> y1
        y1 = self._t1_m1_predict(X)

        # Determine what to correct: y2 from previous round (or y1 if first round)
        y_current = self.m2.predict(y1, gap, missing_now) if self.m2.is_fitted else y1.copy()

        # t=3: y1 --> RM2 --> M2
        self._t3_rm2_update_m2(y_current, y1, X, groups, gap, missing_now)

        # t=2: M2 --> y2 --> RM1  and  h --> M1 --> y1
        y1_t2, y2, rm1_scores = self._t2_m2_predict_and_audit(
            X, groups, gap, missing_now, S
        )

        result = {
            "round":               round_idx,
            "rm1_score":           rm1_scores["rm1_score"],
            "rm2_r2":              self.rm2.last_r2,
            "y1_raw_disparity":    rm1_scores["y1_raw_disparity"],
            "y1_adj_disparity":    rm1_scores["y1_adj_disparity"],
            "y2_raw_disparity":    rm1_scores["y2_raw_disparity"],
            "y2_adj_disparity":    rm1_scores["y2_adj_disparity"],
            "disparity_reduction": rm1_scores["disparity_reduction"],
            "y1_mean_g0":          float(y1[groups == 0].mean()),
            "y1_mean_g1":          float(y1[groups == 1].mean()),
            "y2_mean_g0":          float(y2[groups == 0].mean()),
            "y2_mean_g1":          float(y2[groups == 1].mean()),
        }
        self.history.append(result)
        return result

    def train(self, df_feat: pd.DataFrame, n_rounds: int = 8) -> pd.DataFrame:
        """Run the full causal RLHF training loop for n_rounds.

        Returns a DataFrame with per-round metrics.
        """
        print(f"  Running Causal RLHF pipeline ({n_rounds} rounds) ...")
        print(f"  {'Round':>6}  {'RM1 score':>10}  {'RM2 R²':>8}  "
              f"{'y1 disp':>9}  {'y2 disp':>9}  {'Δ|disp|':>9}")
        print(f"  {'-'*60}")
        for r in range(n_rounds):
            res = self._run_round(df_feat, r)
            print(
                f"  {res['round']:>6}  {res['rm1_score']:>10.4f}  "
                f"{res['rm2_r2']:>8.4f}  "
                f"{res['y1_adj_disparity']:>+9.4f}  "
                f"{res['y2_adj_disparity']:>+9.4f}  "
                f"{res['disparity_reduction']:>+9.4f}"
            )
        return pd.DataFrame(self.history)

    def predict(self, df_feat: pd.DataFrame) -> tuple:
        """Return (y1, y2) for the given feature dataframe."""
        X = df_feat[self.feature_cols].values
        gap = df_feat["gap"].values.astype(int)
        missing_now = df_feat["missing_now"].values.astype(int)
        y1 = self._t1_m1_predict(X)
        y2 = self.m2.predict(y1, gap, missing_now) if self.m2.is_fitted else y1.copy()
        return y1, y2


# ============================================================================
# Simulation wrapper — makes (M1, M2) behave like sklearn predict_proba
# ============================================================================

class CausalRLHFWrapper:
    """Wraps (M1, M2) into a predict_proba-compatible model for run_simulation.

    run_simulation calls model.predict_proba(features)[:,1] where
    features = [x_ffill, missing_now, gap, meas_count]  (FEATURE_COLS order).

    This wrapper feeds those features through M1 first, then passes
    (y1, gap, missing_now) through M2 to obtain debiased predictions y2.
    """

    def __init__(self, m1, m2: DebiasingModel):
        self.m1 = m1
        self.m2 = m2

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # FEATURE_COLS = ["x_ffill", "missing_now", "gap", "meas_count"]
        y1 = self.m1.predict_proba(X)[:, 1]
        gap = X[:, 2].astype(int)          # column 2 = gap
        missing_now = X[:, 1].astype(int)  # column 1 = missing_now
        y2 = self.m2.predict(y1, gap, missing_now)
        return np.column_stack([1.0 - y2, y2])


# ============================================================================
# Top-level entry point: train + simulate + compare
# ============================================================================

def run_causal_rlhf(
    m1_model,
    df_feat: pd.DataFrame,
    sim_kwargs: dict,
    n_rounds: int = 8,
    out_dir: str = "outputs",
) -> dict:
    """Train causal RLHF pipeline and compare it to M1 baseline simulation.

    Parameters
    ----------
    m1_model  : fitted sklearn model (M1, frozen)
    df_feat   : feature dataframe from build_features()
    sim_kwargs: kwargs for run_simulation (T_sim, p11, p01, …)
    n_rounds  : number of causal RLHF training rounds
    out_dir   : directory for saved plots and CSVs

    Returns
    -------
    dict with scalar disparity metrics (baseline vs causal RLHF)
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. Train causal RLHF pipeline ────────────────────────────────────────
    pipeline = CausalRLHFPipeline(m1_model)
    history_df = pipeline.train(df_feat, n_rounds=n_rounds)
    history_df.to_csv(f"{out_dir}/causal_rlhf_training_history.csv", index=False)

    # ── 2. Run simulations ────────────────────────────────────────────────────
    # Baseline: raw M1
    df_base = run_simulation(m1_model, df_feat, **sim_kwargs)

    # Causal RLHF: M2 wrapping M1
    wrapper = CausalRLHFWrapper(m1_model, pipeline.m2)
    df_rlhf = run_simulation(wrapper, df_feat, **sim_kwargs)

    plot_simulation(df_base, label="Baseline (M1)",    out_dir=out_dir, suffix="causal_rlhf_baseline")
    plot_simulation(df_rlhf, label="Causal RLHF (M2)", out_dir=out_dir, suffix="causal_rlhf_mitigated")

    # ── 3. Comparison plot: 3-panel trajectory ────────────────────────────────
    _plot_comparison(df_base, df_rlhf, history_df, out_dir)

    # ── 4. Training-curve plot ────────────────────────────────────────────────
    _plot_training_curves(history_df, out_dir)

    # ── 5. Disparity summary ──────────────────────────────────────────────────
    def _disp(df_c, col):
        g0 = df_c[df_c["group"] == 0][col].mean()
        g1 = df_c[df_c["group"] == 1][col].mean()
        return float(g0 - g1)

    summary = {
        "causal_rlhf_baseline_meas_disparity":  _disp(df_base, "meas_rate"),
        "causal_rlhf_meas_disparity":           _disp(df_rlhf, "meas_rate"),
        "causal_rlhf_baseline_risk_disparity":  _disp(df_base, "mean_pred_risk"),
        "causal_rlhf_risk_disparity":           _disp(df_rlhf, "mean_pred_risk"),
        "causal_rlhf_baseline_gap_disparity":   _disp(df_base, "mean_gap"),
        "causal_rlhf_gap_disparity":            _disp(df_rlhf, "mean_gap"),
        "causal_rlhf_final_rm1_score":          float(history_df["rm1_score"].iloc[-1]),
        "causal_rlhf_final_y2_adj_disparity":   float(history_df["y2_adj_disparity"].iloc[-1]),
        "causal_rlhf_final_y1_adj_disparity":   float(history_df["y1_adj_disparity"].iloc[-1]),
    }

    # CSV summary table
    rows = []
    for cond_lbl, df_c in [("Baseline (M1)", df_base), ("Causal RLHF (M2)", df_rlhf)]:
        for g in [0, 1]:
            last = df_c[(df_c["group"] == g) & (df_c["t"] >= df_c["t"].max() - 4)]
            rows.append({
                "condition": cond_lbl,
                "group": g,
                "mean_meas_rate_last5":  last["meas_rate"].mean(),
                "mean_pred_risk_last5":  last["mean_pred_risk"].mean(),
                "mean_gap_last5":        last["mean_gap"].mean(),
                "true_severity_last5":   last["true_severity"].mean(),
            })
    pd.DataFrame(rows).to_csv(f"{out_dir}/causal_rlhf_sim_summary.csv", index=False)

    return summary


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_comparison(df_base, df_rlhf, history_df, out_dir):
    """3-panel deployment trajectory comparison + RM1 score over rounds."""
    panels = [
        ("meas_rate",      "Measurement Rate"),
        ("mean_pred_risk", "Mean Predicted Risk"),
        ("mean_gap",       "Mean Gap (steps)"),
    ]
    colors_g = {0: "steelblue", 1: "darkorange"}
    ls = {"Baseline (M1)": "-", "Causal RLHF (M2)": "--"}
    alpha = {"Baseline (M1)": 0.50, "Causal RLHF (M2)": 1.0}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (col, title) in zip(axes, panels):
        for cond_lbl, df_c in [("Baseline (M1)", df_base), ("Causal RLHF (M2)", df_rlhf)]:
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
        ax.legend(fontsize=6.5, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Baseline (M1) vs Causal RLHF (M2) — Deployment Simulation",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(f"{out_dir}/causal_rlhf_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_training_curves(history_df, out_dir):
    """Plot per-round training diagnostics for the causal RLHF pipeline."""
    rounds = history_df["round"].values

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: RM1 score over rounds
    axes[0].plot(rounds, history_df["rm1_score"], "o-", color="seagreen", linewidth=2)
    axes[0].set_title("RM1 Fairness Score\n(1 − |severity-adj disparity|)")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("RM1 score")
    axes[0].set_ylim([0, 1.05])
    axes[0].axhline(1.0, color="gray", linestyle=":", linewidth=1)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: y1 vs y2 adjusted disparity over rounds
    axes[1].plot(rounds, history_df["y1_adj_disparity"], "s--",
                 color="crimson",  linewidth=2, label="y1 (M1, frozen)")
    axes[1].plot(rounds, history_df["y2_adj_disparity"], "o-",
                 color="seagreen", linewidth=2, label="y2 (M2, debiased)")
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle=":")
    axes[1].set_title("Severity-Adjusted Disparity\n(G0 − G1; 0 = perfectly fair)")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Adjusted disparity")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: mean predictions by group for y1 and y2
    axes[2].plot(rounds, history_df["y1_mean_g0"], "s--", color="steelblue",  linewidth=2, label="y1 G0")
    axes[2].plot(rounds, history_df["y1_mean_g1"], "s--", color="darkorange", linewidth=2, label="y1 G1")
    axes[2].plot(rounds, history_df["y2_mean_g0"], "o-",  color="steelblue",  linewidth=2, label="y2 G0", alpha=0.7)
    axes[2].plot(rounds, history_df["y2_mean_g1"], "o-",  color="darkorange", linewidth=2, label="y2 G1", alpha=0.7)
    axes[2].set_title("Mean Predicted Risk by Group\n(dashed=y1/M1, solid=y2/M2)")
    axes[2].set_xlabel("Round")
    axes[2].set_ylabel("Mean predicted risk")
    axes[2].legend(fontsize=7, ncol=2)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        "Causal RLHF Training Diagnostics\n"
        "t=1: x→M1→y1  |  t=3: y1→RM2→M2  |  t=2: M2→y2→RM1, h→M1→y1",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(f"{out_dir}/causal_rlhf_training_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    from toy.generate_data import generate_data
    from toy.train_model import build_features, train_model

    df = generate_data()
    df_feat = build_features(df)
    model, metrics, _ = train_model(df_feat)

    sim_kwargs = dict(
        T_sim=30, p11=0.8, p01=0.2, mu0=0.0, mu1=1.0,
        sigma=1.0, tau=0.5, p_high=0.85, p_low=0.15, seed=123,
    )
    summary = run_causal_rlhf(model, df_feat, sim_kwargs)
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")
