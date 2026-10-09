import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

FEATURES = ["ret", "vol_21", "vix"]


def _as_feature_frame(feats):
    """Normalize feature input to a DataFrame with the expected columns."""
    if isinstance(feats, pd.DataFrame):
        df = feats.copy()
    elif isinstance(feats, np.ndarray):
        if feats.ndim != 2:
            raise ValueError("feats must be a 2D array or DataFrame")
        if feats.shape[1] != len(FEATURES):
            raise ValueError(
                f"Expected {len(FEATURES)} feature columns, got {feats.shape[1]}."
            )
        df = pd.DataFrame(feats, columns=FEATURES)
    else:
        raise TypeError("feats must be a pandas DataFrame or 2D NumPy array")

    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")

    return df[FEATURES].astype(float).dropna()


def fit_hmm(feats, n_states=3, n_init=10, seed=42):
    """Fit a Gaussian HMM, trying several random starts and keeping the best."""
    X_df = _as_feature_frame(feats)
    if len(X_df) == 0:
        raise ValueError("No usable rows remain after dropping missing values.")
    if n_states < 1:
        raise ValueError("n_states must be at least 1.")

    scaler = StandardScaler()
    X = scaler.fit_transform(X_df[FEATURES])

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

    if best_model is None:
        raise ValueError("Model fitting failed; no valid HMM was trained.")

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
    df = _as_feature_frame(feats)
    df["state"] = states
    summary = df.groupby("state").mean()
    summary["days"] = df.groupby("state").size()
    return summary


def count_params(n_states, n_features):
    """Free parameters in a full-covariance Gaussian HMM."""
    start = n_states - 1
    trans = n_states * (n_states - 1)
    means = n_states * n_features
    covs = n_states * n_features * (n_features + 1) / 2
    return start + trans + means + covs


def select_n_states(feats, state_range=range(2, 6)):
    """Fit HMMs with different numbers of states and compare AIC/BIC."""
    state_list = list(state_range)
    if not state_list:
        raise ValueError("state_range must contain at least one state count.")
    if any(k < 1 for k in state_list):
        raise ValueError("state_range values must be positive integers.")

    rows = []
    for k in state_list:
        model, _, X = fit_hmm(feats, n_states=k)
        log_lik = model.score(X)
        p = count_params(k, X.shape[1])
        n = len(X)
        rows.append({
            "n_states": k,
            "log_likelihood": log_lik,
            "AIC": 2 * p - 2 * log_lik,
            "BIC": p * np.log(n) - 2 * log_lik,
        })
    return pd.DataFrame(rows).set_index("n_states")