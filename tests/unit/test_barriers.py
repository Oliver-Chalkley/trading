import pandas as pd
import pytest

from trading.data import load_ohlcv
from trading.features.volatility import ewm_volatility
from trading.labeling.barriers import triple_barrier_labels

_VALID_LABELS = {-1, 0, 1}


def _make_close(values: list[float]) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx, name="close")


def _make_vol(values: list[float]) -> pd.Series:
    """Synthetic volatility series with leading NaNs matching length."""
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx, name="ewm_vol")


# --- output metadata ---


def test_output_type():
    close = _make_close([100.0 + i for i in range(30)])
    vol = ewm_volatility(close, span=5)
    result = triple_barrier_labels(close, vol)
    assert isinstance(result, pd.Series)


def test_output_index():
    close = _make_close([100.0 + i for i in range(30)])
    vol = ewm_volatility(close, span=5)
    result = triple_barrier_labels(close, vol)
    pd.testing.assert_index_equal(result.index, close.index)


def test_output_name():
    close = _make_close([100.0 + i for i in range(30)])
    vol = ewm_volatility(close, span=5)
    result = triple_barrier_labels(close, vol)
    assert result.name == "label"


# --- label validity ---


def test_labels_are_valid():
    close = _make_close([100.0 + i * 0.3 for i in range(50)])
    vol = ewm_volatility(close, span=5)
    result = triple_barrier_labels(close, vol)
    non_nan = result.dropna()
    assert set(non_nan.unique()).issubset(_VALID_LABELS)


def test_nan_where_vol_nan():
    """Labels must be NaN wherever volatility is NaN (warm-up period)."""
    span = 5
    close = _make_close([100.0 + i for i in range(30)])
    vol = ewm_volatility(close, span=span)
    result = triple_barrier_labels(close, vol, max_hold=3)
    # vol warm-up: first span-1 values are NaN
    assert result.iloc[: span - 1].isna().all()


# --- barrier logic ---


def test_uptrend_hits_pt():
    """Strong uptrend with generous PT and tight SL → most labels +1."""
    n = 100
    # Prices rise 5% per bar — easily hits profit-take
    close = _make_close([100.0 * (1.05**i) for i in range(n)])
    vol = _make_vol([float("nan")] * 4 + [0.01] * (n - 4))
    result = triple_barrier_labels(close, vol, pt_sl=(1.0, 3.0), max_hold=5)
    non_nan = result.dropna()
    assert (non_nan == 1).sum() > len(non_nan) // 2  # noqa: PLR2004


def test_downtrend_hits_sl():
    """Strong downtrend with generous SL and tight PT → most labels -1."""
    n = 100
    # Prices fall 5% per bar — easily hits stop-loss
    close = _make_close([100.0 * (0.95**i) for i in range(n)])
    vol = _make_vol([float("nan")] * 4 + [0.01] * (n - 4))
    result = triple_barrier_labels(close, vol, pt_sl=(3.0, 1.0), max_hold=5)
    non_nan = result.dropna()
    assert (non_nan == -1).sum() > len(non_nan) // 2  # noqa: PLR2004


def test_flat_hits_time():
    """Flat prices → barriers never hit → all non-NaN labels are 0."""
    n = 40
    close = _make_close([100.0] * n)
    vol = _make_vol([float("nan")] * 4 + [0.01] * (n - 4))
    result = triple_barrier_labels(close, vol, pt_sl=(1.0, 1.0), max_hold=5)
    non_nan = result.dropna()
    assert (non_nan == 0).all()


def test_pt_zero_no_positive_labels():
    """`pt=0` disables upper barrier → no +1 labels possible."""
    n = 50
    close = _make_close([100.0 * (1.03**i) for i in range(n)])
    vol = _make_vol([float("nan")] * 4 + [0.01] * (n - 4))
    result = triple_barrier_labels(close, vol, pt_sl=(0.0, 1.0), max_hold=5)
    assert (result.dropna() == 1).sum() == 0


def test_sl_zero_no_negative_labels():
    """`sl=0` disables lower barrier → no -1 labels possible."""
    n = 50
    close = _make_close([100.0 * (0.97**i) for i in range(n)])
    vol = _make_vol([float("nan")] * 4 + [0.01] * (n - 4))
    result = triple_barrier_labels(close, vol, pt_sl=(1.0, 0.0), max_hold=5)
    assert (result.dropna() == -1).sum() == 0


def test_max_hold_zero():
    """`max_hold=0` → no forward window → all non-NaN labels are 0."""
    n = 30
    close = _make_close([100.0 * (1.05**i) for i in range(n)])
    vol = _make_vol([float("nan")] * 4 + [0.01] * (n - 4))
    result = triple_barrier_labels(close, vol, pt_sl=(1.0, 1.0), max_hold=0)
    assert (result.dropna() == 0).all()


# --- edge cases ---


def test_near_end_no_nan():
    """Bars near end of series (truncated window) should return 0, not NaN."""
    n = 20
    close = _make_close([100.0] * n)
    vol = _make_vol([0.01] * n)
    result = triple_barrier_labels(close, vol, pt_sl=(1.0, 1.0), max_hold=10)
    # Last bar has empty window — should be 0 (time barrier), not NaN
    assert not pd.isna(result.iloc[-1])
    assert result.iloc[-1] == 0


def test_non_positive_prices_raises():
    close = _make_close([100.0, 0.0, 102.0])
    vol = _make_vol([0.01] * 3)
    with pytest.raises(ValueError):
        triple_barrier_labels(close, vol)


# --- integration smoke test ---


def test_real_data_smoke():
    df = load_ohlcv()
    close: pd.Series = df["close"]  # type: ignore[assignment]
    vol = ewm_volatility(close)
    result = triple_barrier_labels(close, vol, pt_sl=(2.0, 1.0), max_hold=20)
    assert len(result) == len(close)
    assert result.name == "label"
    non_nan = result.dropna()
    assert set(non_nan.unique()).issubset(_VALID_LABELS)
    assert non_nan.notna().all()
