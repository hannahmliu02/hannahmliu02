"""
train_model.py
--------------
Build per-(patient, t) features from the raw trajectory data and train a
logistic-regression risk model to predict the patient outcome Y.

Feature engineering:
  - x_ffill      : last observed X value (forward-filled per patient; filled
                   with global mean if never observed before current step)
  - missing_now  : 1 if X is NaN at current timestep (missingness indicator)
  - gap           : timesteps since last measurement (0 if measured now)
  - meas_count   : cumulative number of measurements so far (inclusive)

The missingness indicator and gap are the 'shortcut' features the model can
exploit — if being measured correlates with severity, the model can learn
"unmeasured ≈ not sick" even when that's due to subgroup bias.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the trajectory dataframe.

    Operates per patient (sorted by t) so that forward-fill and gap
    calculations respect temporal ordering.
    """
    records = []
    global_mean_x = df["X"].mean()  # fallback when X never observed yet

    for pid, grp in df.groupby("patient_id", sort=False):
        grp = grp.sort_values("t").copy()
        last_obs = np.nan
        gap = 0
        count = 0

        for _, row in grp.iterrows():
            if not np.isnan(row["X"]):
                last_obs = row["X"]
                gap = 0
                count += 1
            else:
                gap += 1

            x_ffill = last_obs if not np.isnan(last_obs) else global_mean_x
            missing_now = int(np.isnan(row["X"]))

            records.append(
                {
                    "patient_id": row["patient_id"],
                    "t": row["t"],
                    "group": row["group"],
                    "S": row["S"],
                    "M": row["M"],
                    "X": row["X"],
                    "Y": row["Y"],
                    "x_ffill": x_ffill,
                    "missing_now": missing_now,
                    "gap": gap,
                    "meas_count": count,
                }
            )

    return pd.DataFrame(records)


FEATURE_COLS = ["x_ffill", "missing_now", "gap", "meas_count"]


def train_model(
    df_feat: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple:
    """Train logistic regression on the feature dataframe.

    Returns
    -------
    model : fitted sklearn estimator
    metrics : dict with AUROC and calibration curve arrays
    df_test : test-set dataframe with added 'pred_prob' column
    """
    X = df_feat[FEATURE_COLS].values
    y = df_feat["Y"].values

    # Stratified split so both classes appear in train/test
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df_feat.index, test_size=test_size, random_state=seed, stratify=y
    )

    model = LogisticRegression(max_iter=500, random_state=seed)
    model.fit(X_tr, y_tr)

    y_prob = model.predict_proba(X_te)[:, 1]
    auroc = roc_auc_score(y_te, y_prob)

    # Calibration curve
    fraction_pos, mean_pred = calibration_curve(y_te, y_prob, n_bins=10)

    # Wald significance via statsmodels (same training split, no regularisation)
    sig_df = significance_report(X_tr, y_tr, FEATURE_COLS)

    metrics = {
        "auroc": auroc,
        "calibration_fraction_pos": fraction_pos.tolist(),
        "calibration_mean_pred": mean_pred.tolist(),
        "feature_coefs": dict(zip(FEATURE_COLS, model.coef_[0].tolist())),
        # statsmodels Wald table — serialisable list of dicts
        "coef_significance": sig_df.to_dict(orient="records"),
    }

    df_test = df_feat.loc[idx_te].copy()
    df_test["pred_prob"] = y_prob

    return model, metrics, df_test


def significance_report(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    feature_names: list,
) -> pd.DataFrame:
    """Fit logistic regression via statsmodels to get Wald significance tests.

    scikit-learn's LogisticRegression maximises a regularised objective and
    does not provide standard errors.  statsmodels.Logit uses maximum
    likelihood without regularisation and computes standard errors from the
    observed Fisher information matrix, giving exact Wald z-statistics and
    two-sided p-values.

    The intercept is prepended automatically via sm.add_constant().

    Parameters
    ----------
    X_tr : training feature matrix (same split used for sklearn fit)
    y_tr : training labels
    feature_names : list of feature column names (without intercept)

    Returns
    -------
    pd.DataFrame with one row per parameter:
        feature | coef | std_err | z | pvalue | ci_025 | ci_975 | stars
    """
    X_const = sm.add_constant(X_tr, has_constant="add")
    names   = ["intercept"] + list(feature_names)

    result = sm.Logit(y_tr, X_const).fit(disp=False)  # disp=False → no iteration log

    ci = result.conf_int()  # shape (n_params, 2)

    def _stars(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        if p < 0.10:  return "."
        return "ns"

    rows = []
    for i, feat in enumerate(names):
        rows.append({
            "feature": feat,
            "coef":    float(result.params[i]),
            "std_err": float(result.bse[i]),
            "z":       float(result.tvalues[i]),
            "pvalue":  float(result.pvalues[i]),
            "ci_025":  float(ci[i, 0]),
            "ci_975":  float(ci[i, 1]),
            "stars":   _stars(result.pvalues[i]),
        })
    return pd.DataFrame(rows)


def save_model(model, path: str = "outputs/model.pkl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str = "outputs/model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    from toy.generate_data import generate_data

    df = generate_data()
    df_feat = build_features(df)
    model, metrics, df_test = train_model(df_feat)

    print(f"AUROC: {metrics['auroc']:.4f}")
    print("Feature coefficients:")
    for feat, coef in metrics["feature_coefs"].items():
        print(f"  {feat:15s}: {coef:+.4f}")
