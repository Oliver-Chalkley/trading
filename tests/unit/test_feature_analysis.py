import numpy as np
import pandas as pd
import pytest

from trading.data import load_ohlcv
from trading.features import atr, bollinger_bands, macd, rsi, volume_ratio
from trading.features.analysis import (
    adf_stationarity,
    autocorrelation,
    correlation_matrix,
    feature_summary,
)


def _make_prices(values: list[float]) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx, name="close")


def _make_feature_df(n: int = 60) -> pd.DataFrame:
    """Build a small feature DataFrame from synthetic prices with variation."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    returns = rng.standard_normal(n) * 0.01
    close = pd.Series(100.0 * np.exp(returns.cumsum()), index=idx)
    high = close * 1.01
    low = close * 0.99
    return pd.DataFrame(
        {
            "rsi": rsi(close),
            "atr": atr(high, low, close),
        }
    ).dropna()


# --- feature_summary ---


def test_summary_type():
    df = _make_feature_df()
    result = feature_summary(df)
    assert isinstance(result, pd.DataFrame)


def test_summary_rows():
    df = _make_feature_df()
    result = feature_summary(df)
    assert list(result.index) == ["mean", "std", "skew", "kurtosis", "min", "max"]


def test_summary_columns():
    df = _make_feature_df()
    result = feature_summary(df)
    assert list(result.columns) == list(df.columns)


def test_summary_mean_correct():
    df = _make_feature_df()
    result = feature_summary(df)
    pd.testing.assert_series_equal(
        result.loc["mean"],
        df.mean(),
        check_names=False,
    )


def test_summary_empty_raises():
    with pytest.raises(ValueError):
        feature_summary(pd.DataFrame())


# --- autocorrelation ---


def test_ac_type():
    series = _make_prices([float(i) for i in range(30)])
    result = autocorrelation(series, lags=[1, 2, 3])
    assert isinstance(result, pd.Series)


def test_ac_name():
    series = _make_prices([float(i) for i in range(30)])
    result = autocorrelation(series, lags=[1, 2])
    assert result.name == "autocorrelation"


def test_ac_index():
    series = _make_prices([float(i) for i in range(30)])
    lags = [1, 5, 10]
    result = autocorrelation(series, lags=lags)
    assert list(result.index) == lags


def test_ac_lag0_is_one():
    series = _make_prices([float(i) for i in range(30)])
    result = autocorrelation(series, lags=[0])
    assert abs(result.iloc[0] - 1.0) < 1e-9  # noqa: PLR2004


def test_ac_short_series_raises():
    series = _make_prices([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        autocorrelation(series, lags=[5])


# --- adf_stationarity ---


def test_adf_keys():
    rng = np.random.default_rng(42)
    series = _make_prices(rng.standard_normal(50).tolist())
    result = adf_stationarity(series)
    assert set(result.keys()) == {"stat", "p_value", "is_stationary"}


def test_adf_stationary_series():
    """White noise should be stationary."""
    rng = np.random.default_rng(42)
    series = _make_prices(rng.standard_normal(200).tolist())
    result = adf_stationarity(series)
    assert result["is_stationary"] is True


def test_adf_nonstationary_series():
    """Random walk (cumulative sum of noise) should be non-stationary."""
    rng = np.random.default_rng(0)
    walk = rng.standard_normal(500).cumsum().tolist()
    series = _make_prices(walk)
    result = adf_stationarity(series)
    assert result["is_stationary"] is False


def test_adf_too_short_raises():
    series = _make_prices([float(i) for i in range(10)])
    with pytest.raises(ValueError):
        adf_stationarity(series)


# --- correlation_matrix ---


def test_corr_type():
    df = _make_feature_df()
    result = correlation_matrix(df)
    assert isinstance(result, pd.DataFrame)


def test_corr_diagonal():
    df = _make_feature_df()
    result = correlation_matrix(df)
    for col in result.columns:
        assert abs(result.loc[col, col] - 1.0) < 1e-9  # noqa: PLR2004


def test_corr_symmetric():
    df = _make_feature_df()
    result = correlation_matrix(df)
    pd.testing.assert_frame_equal(result, result.T)


def test_corr_single_column_raises():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        correlation_matrix(df)


def test_corr_empty_raises():
    with pytest.raises(ValueError):
        correlation_matrix(pd.DataFrame())


# --- Integration smoke test ---


def test_real_data_smoke():
    df = load_ohlcv()
    close: pd.Series = df["close"]  # type: ignore[assignment]
    high: pd.Series = df["high"]  # type: ignore[assignment]
    low: pd.Series = df["low"]  # type: ignore[assignment]
    volume: pd.Series = df["volume"]  # type: ignore[assignment]

    features = pd.DataFrame(
        {
            "rsi": rsi(close),
            "atr": atr(high, low, close),
            "volume_ratio": volume_ratio(volume),
        }
    ).dropna()

    bb = bollinger_bands(close).dropna()
    features = features.join(bb, how="inner")
    macd_df = macd(close).dropna()
    features = features.join(macd_df, how="inner")

    summary = feature_summary(features)
    assert summary.shape == (6, len(features.columns))

    rsi_col: pd.Series = features["rsi"]  # type: ignore[assignment]
    ac = autocorrelation(rsi_col, lags=[1, 5, 10])
    assert len(ac) == 3  # noqa: PLR2004

    adf = adf_stationarity(rsi_col)
    assert "is_stationary" in adf

    corr = correlation_matrix(features)
    assert corr.shape == (len(features.columns), len(features.columns))
