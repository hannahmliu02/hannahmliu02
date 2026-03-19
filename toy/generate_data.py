"""
generate_data.py
----------------
Simulate N patients over T timesteps with an informative (biased) observation
process.  The key mechanism:

  - Each patient has a hidden severity state S_t ∈ {0,1} (Markov chain).
  - Measurements M_t are more likely when S_t=1 (clinician tendency to monitor
    sick patients), BUT subgroup g=1 patients are systematically under-measured
    even at the same severity — mimicking real-world access / cost disparities.
  - Observed feature X_t is drawn from N(mu1, sigma) when S_t=1 and
    N(mu0, sigma) when S_t=0; it is NaN when M_t=0.

Returns a tidy DataFrame: patient_id, t, group, S, M, X
"""

import numpy as np
import pandas as pd


def generate_data(
    N: int = 2000,
    T: int = 20,
    p11: float = 0.8,   # P(S_{t+1}=1 | S_t=1)
    p01: float = 0.2,   # P(S_{t+1}=1 | S_t=0)
    # Measurement probabilities [group0, group1] × [S=0, S=1]
    meas_prob_s0: tuple = (0.30, 0.15),  # P(M=1 | S=0) per group
    meas_prob_s1: tuple = (0.80, 0.60),  # P(M=1 | S=1) per group
    mu0: float = 0.0,
    mu1: float = 1.0,
    sigma: float = 1.0,
    noise_flip: float = 0.05,  # label-noise probability for Y
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic patient trajectories.

    Parameters
    ----------
    N : total number of patients
    T : number of timesteps per patient
    p11, p01 : Markov transition probabilities for severity state
    meas_prob_s0/s1 : tuple (group0_prob, group1_prob) for measurement
    mu0, mu1 : mean of observed feature when S=0/1
    sigma : std of observed feature
    noise_flip : probability of flipping the outcome label (adds noise so
                 Y is not perfectly predictable, giving ~50% prevalence
                 and making tau=0.5 a meaningful policy threshold)
    seed : random seed for reproducibility

    Outcome definition
    ------------------
    Y = S at the *final* timestep (T-1), with noise_flip label noise.
    Using the current-step severity (~50% prevalence) rather than a rolling
    window ensures predicted risks are centred near 0.5, making the
    risk-threshold policy actually discriminate between patients.

    Returns
    -------
    pd.DataFrame with columns:
        patient_id, t, group, S, M, X, Y
    """
    rng = np.random.default_rng(seed)

    records = []

    for pid in range(N):
        # Assign subgroup: half group 0, half group 1
        g = int(pid >= N // 2)

        # Initial severity state (stationary distribution of Markov chain)
        pi1 = p01 / (1 - p11 + p01)          # stationary P(S=1)
        S = int(rng.random() < pi1)

        severity_history = []

        for t in range(T):
            # -- Measurement decision --
            if S == 1:
                m_prob = meas_prob_s1[g]
            else:
                m_prob = meas_prob_s0[g]
            M = int(rng.random() < m_prob)

            # -- Observed feature --
            if M == 1:
                mu = mu1 if S == 1 else mu0
                X = float(rng.normal(mu, sigma))
            else:
                X = float("nan")

            severity_history.append(S)
            records.append((pid, t, g, S, M, X))

            # -- Severity transition --
            if S == 1:
                S = int(rng.random() < p11)
            else:
                S = int(rng.random() < p01)

        # -- Outcome label: Y = severity at the final step, with label noise --
        # Using the last recorded severity (before the final transition above)
        final_severity = severity_history[-1]
        Y = int(final_severity == 1)
        if rng.random() < noise_flip:
            Y = 1 - Y   # flip with small probability

        # Attach Y to all rows for this patient (broadcast)
        for i in range(len(records) - T, len(records)):
            records[i] = records[i] + (Y,)

    df = pd.DataFrame(records, columns=["patient_id", "t", "group", "S", "M", "X", "Y"])
    return df


if __name__ == "__main__":
    df = generate_data()
    print(df.head(20))
    print(f"\nShape: {df.shape}")
    print(f"Measurement rate by group:\n{df.groupby('group')['M'].mean()}")
    print(f"Severity rate by group:\n{df.groupby('group')['S'].mean()}")
    print(f"Outcome prevalence by group:\n{df.groupby('group')['Y'].mean()}")
