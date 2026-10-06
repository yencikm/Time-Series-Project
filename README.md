# Regime-Switching Market Model

Detecting market regimes (calm, volatile, crisis) with a Hidden Markov Model
and forecasting volatility conditional on the current regime.

## Motivation
Markets don't behave the same way all the time. ...

## Data
S&P 500 daily prices (2000 to present) via yfinance.

## Methodology
1. Feature engineering (returns, realized volatility, momentum)
2. Gaussian HMM for regime detection
3. Regime-conditional forecasting vs. baselines (GARCH)
4. Walk-forward evaluation

## Results
*Coming soon*

## How to run
pip install -r requirements.txt
python -m src.data

## Limitations
*Coming soon*
