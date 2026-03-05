import pathlib

import pandas as pd

DEFAULT_PATH = pathlib.Path(__file__).parent.parent.parent / "data" / "spy_daily.csv"


def load_ohlcv(path: pathlib.Path | str = DEFAULT_PATH) -> pd.DataFrame:
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    df = df[["open", "high", "low", "close", "volume"]]
    df.index.name = "timestamp"
    df.sort_index(inplace=True)
    return df
