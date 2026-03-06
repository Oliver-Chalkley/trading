import pandas as pd
import pytest

from trading.data import load_ohlcv
from trading.features.volatility import ewm_volatility


def _make_prices(values: list[float]) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx, name="close")


# --- output shape / metadata ---


def test_output_type():
    prices = _make_prices([100.0, 101.0, 102.0, 103.0, 104.0])
    result = ewm_volatility(prices)
    assert isinstance(result, pd.Series)


def test_output_index():
    prices = _make_prices([100.0, 101.0, 102.0, 103.0, 104.0])
    result = ewm_volatility(prices)
    pd.testing.assert_index_equal(result.index, prices.index)


def test_output_name():
    prices = _make_prices([100.0, 101.0, 102.0, 103.0, 104.0])
    result = ewm_volatility(prices)
    assert result.name == "ewm_vol"


# --- NaN behaviour ---


def test_leading_nans():
    span = 5
    prices = _make_prices([100.0 + i for i in range(20)])
    result = ewm_volatility(prices, span=span)
    # first span-1 values should be NaN (min_periods=span)
    assert result.iloc[: span - 1].isna().all()
    assert result.iloc[span:].notna().all()


def test_single_row():
    prices = _make_prices([100.0])
    result = ewm_volatility(prices)
    assert result.isna().all()


# --- value correctness ---


def test_non_negative():
    prices = _make_prices([100.0 * (1.01**i) for i in range(50)])
    result = ewm_volatility(prices, span=10)
    assert (result.dropna() >= 0).all()


def test_constant_prices():
    prices = _make_prices([100.0] * 30)
    result = ewm_volatility(prices, span=5)
    non_nan = result.dropna()
    assert (non_nan == 0).all()


def test_custom_span_warmup():
    """Smaller span warms up sooner (fewer leading NaNs)."""
    prices = _make_prices([100.0 + i for i in range(50)])
    result_fast = ewm_volatility(prices, span=5)
    result_slow = ewm_volatility(prices, span=20)
    assert result_fast.notna().sum() > result_slow.notna().sum()


# --- validation ---


def test_non_positive_prices_raises():
    prices = _make_prices([100.0, 0.0, 102.0])
    with pytest.raises(ValueError):
        ewm_volatility(prices)


def test_negative_prices_raises():
    prices = _make_prices([100.0, -1.0, 102.0])
    with pytest.raises(ValueError):
        ewm_volatility(prices)


# --- integration smoke test ---


def test_real_data():
    df = load_ohlcv()
    close: pd.Series = df["close"]  # type: ignore[assignment]
    result = ewm_volatility(close)
    assert len(result) == len(df)
    assert result.notna().any()
