import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

FEATURES = ["ret", "vol_21", "vix"]

def fit_hmm(feats, n_states=3, n_init=10, seed=42):
    """Fit a Gaussian HMM, trying several random starts and keeping the best."""
    scaler = StandardScaler()
    X = scaler.fit_transform(feats[FEATURES])

    best_model, best_score = None, -np.inf
    for i in range(n_init):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=500,
            random_state=seed + i,
        )
        model.fit(X)
        score = model.score(X)
        if score > best_score:
            best_model, best_score = model, score

    return best_model, scaler, X

def decode_regimes(model, X, index):
    """Return the most likely regime for each day plus regime probabilities."""
    states = pd.Series(model.predict(X), index=index, name="state")
    probs = pd.DataFrame(
        model.predict_proba(X),
        index=index,
        columns=[f"p_state_{i}" for i in range(model.n_components)],
    )
    return states, probs

def summarize_regimes(feats, states):
    """Average behavior of each regime, used to give them names."""
    df = feats[FEATURES].copy()
    df["state"] = states
    summary = df.groupby("state").mean()
    summary["days"] = df.groupby("state").size()
    return summary
