import yfinance as yf
import pandas as pd

def download_prices(ticker="^GSPC", start="2000-01-01", end=None):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)
    return df[["Close"]].rename(columns={"Close": "close"})

def load_data(start="2000-01-01"):
    spx = download_prices("^GSPC", start).rename(columns={"close": "spx"})
    vix = download_prices("^VIX", start).rename(columns={"close": "vix"})
    return spx.join(vix, how="inner").dropna()

if __name__ == "__main__":
    df = load_data()
    df.to_csv("data/market_data.csv")
    print(df.tail())
