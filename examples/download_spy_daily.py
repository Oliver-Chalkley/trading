"""Download SPY daily OHLCV bars from Yahoo Finance (1993-present).

SPY (S&P 500 ETF) launched 1993-01-22, so that is the earliest available date.
Output is saved to data/spy_daily.csv with columns:
    timestamp, open, high, low, close, volume
"""

import pathlib

import pandas as pd
import yfinance as yf

OUTPUT_PATH = pathlib.Path("data/spy_daily.csv")
TICKER = "SPY"
START = "1993-01-01"


def download() -> pd.DataFrame:
    raw = yf.download(TICKER, start=START, interval="1d", auto_adjust=True, progress=True)

    # yfinance returns a MultiIndex when downloading a single ticker with auto_adjust
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    raw.index.name = "timestamp"
    raw.columns = [c.lower() for c in raw.columns]

    df = raw[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index)
    return df


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {TICKER} daily bars from {START}...")
    df = download()

    df.to_csv(OUTPUT_PATH)
    print(f"Saved {len(df):,} rows to {OUTPUT_PATH}")
    print(df.tail())


if __name__ == "__main__":
    main()
