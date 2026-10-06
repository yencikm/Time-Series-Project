import numpy as np
import pandas as pd

def build_features(df):
    out = pd.DataFrame(index=df.index)

    # Daily log return
    out["ret"] = np.log(df["spx"]).diff()

    # Realized volatility (annualized) over short and medium windows
    out["vol_5"] = out["ret"].rolling(5).std() * np.sqrt(252)
    out["vol_21"] = out["ret"].rolling(21).std() * np.sqrt(252)

    # Momentum: return over the past 21 and 63 trading days
    out["mom_21"] = np.log(df["spx"]).diff(21)
    out["mom_63"] = np.log(df["spx"]).diff(63)

    # VIX level and daily change
    out["vix"] = df["vix"]
    out["vix_chg"] = df["vix"].pct_change()

    # Drawdown from the running peak
    out["drawdown"] = df["spx"] / df["spx"].cummax() - 1

    return out.dropna()
