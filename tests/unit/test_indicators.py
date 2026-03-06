import pandas as pd

from trading.data import load_ohlcv
from trading.features.indicators import atr, bollinger_bands, macd, rsi, volume_ratio


def _make_prices(values: list[float]) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx, name="close")


def _make_ohlcv(n: int = 50) -> pd.DataFrame:
    """Simple synthetic OHLCV with trending close prices."""
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    close = pd.Series([100.0 + i for i in range(n)], index=idx)
    high = close + 1.0
    low = close - 1.0
    volume = pd.Series([1_000_000.0] * n, index=idx)
    return pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})


# --- RSI ---


def test_rsi_type():
    prices = _make_prices([100.0 + i for i in range(30)])
    assert isinstance(rsi(prices), pd.Series)


def test_rsi_name():
    prices = _make_prices([100.0 + i for i in range(30)])
    assert rsi(prices).name == "rsi"


def test_rsi_range():
    prices = _make_prices([100.0 + i * 0.5 for i in range(50)])
    result = rsi(prices)
    non_nan = result.dropna()
    assert (non_nan >= 0).all() and (non_nan <= 100).all()  # noqa: PLR2004,PT018


def test_rsi_leading_nans():
    period = 14
    prices = _make_prices([100.0 + i for i in range(40)])
    result = rsi(prices, period=period)
    assert result.iloc[:period].isna().all()
    assert result.iloc[period:].notna().all()


def test_rsi_constant_prices():
    """Constant prices have zero gain and loss; RSI should be NaN or 50."""
    prices = _make_prices([100.0] * 30)
    result = rsi(prices)
    non_nan = result.dropna()
    # acceptable: all NaN or all 50 (implementation-dependent edge case)
    assert non_nan.empty or (non_nan == 50).all()  # noqa: PLR2004,PT018


# --- ATR ---


def _hlc(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    h: pd.Series = df["high"]  # type: ignore[assignment]
    lo: pd.Series = df["low"]  # type: ignore[assignment]
    c: pd.Series = df["close"]  # type: ignore[assignment]
    return h, lo, c


def test_atr_type():
    df = _make_ohlcv()
    assert isinstance(atr(*_hlc(df)), pd.Series)


def test_atr_name():
    df = _make_ohlcv()
    assert atr(*_hlc(df)).name == "atr"


def test_atr_non_negative():
    df = _make_ohlcv()
    result = atr(*_hlc(df))
    assert (result.dropna() >= 0).all()


def test_atr_leading_nans():
    period = 14
    df = _make_ohlcv(50)
    result = atr(*_hlc(df), period=period)
    # tr[0] is non-NaN (pandas max ignores NaN), so warm-up ends at period-1
    assert result.iloc[: period - 1].isna().all()
    assert result.iloc[period - 1 :].notna().all()


# --- MACD ---


def test_macd_type():
    prices = _make_prices([100.0 + i for i in range(60)])
    assert isinstance(macd(prices), pd.DataFrame)


def test_macd_columns():
    prices = _make_prices([100.0 + i for i in range(60)])
    result = macd(prices)
    assert list(result.columns) == ["macd", "signal", "histogram"]


def test_macd_histogram():
    prices = _make_prices([100.0 + i for i in range(60)])
    result = macd(prices)
    pd.testing.assert_series_equal(
        result["histogram"],
        result["macd"] - result["signal"],
        check_names=False,
    )


# --- Bollinger Bands ---


def test_bb_type():
    prices = _make_prices([100.0 + i for i in range(30)])
    assert isinstance(bollinger_bands(prices), pd.DataFrame)


def test_bb_columns():
    prices = _make_prices([100.0 + i for i in range(30)])
    result = bollinger_bands(prices)
    assert list(result.columns) == ["upper", "middle", "lower"]


def test_bb_ordering():
    prices = _make_prices([100.0 + i * 0.3 for i in range(40)])
    result = bollinger_bands(prices).dropna()
    assert (result["upper"] >= result["middle"]).all()
    assert (result["middle"] >= result["lower"]).all()


# --- Volume Ratio ---


def _vol(df: pd.DataFrame) -> pd.Series:
    return df["volume"]  # type: ignore[return-value]


def test_vr_type():
    df = _make_ohlcv()
    assert isinstance(volume_ratio(_vol(df)), pd.Series)


def test_vr_name():
    df = _make_ohlcv()
    assert volume_ratio(_vol(df)).name == "volume_ratio"


def test_vr_constant_volume():
    df = _make_ohlcv(30)
    result = volume_ratio(_vol(df))
    non_nan = result.dropna()
    assert ((non_nan - 1.0).abs() < 1e-9).all()  # noqa: PLR2004


# --- Integration smoke test ---


def test_real_data_smoke():
    df = load_ohlcv()
    close: pd.Series = df["close"]  # type: ignore[assignment]
    high: pd.Series = df["high"]  # type: ignore[assignment]
    low: pd.Series = df["low"]  # type: ignore[assignment]
    volume: pd.Series = df["volume"]  # type: ignore[assignment]
    assert len(rsi(close)) == len(df)
    assert len(atr(high, low, close)) == len(df)
    assert len(macd(close)) == len(df)
    assert len(bollinger_bands(close)) == len(df)
    assert len(volume_ratio(volume)) == len(df)
