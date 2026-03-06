from typing import cast

import pandas as pd


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute Wilder's Relative Strength Index.

    Args:
        prices: Price series with a DatetimeIndex.
        period: Lookback period. Defaults to 14.

    Returns:
        Series of RSI values (0–100) named "rsi". First `period` values are NaN.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    result = cast(pd.Series, 100 - 100 / (1 + rs))
    result.name = "rsi"
    return result


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute Average True Range using Wilder's smoothing.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        period: Lookback period. Defaults to 14.

    Returns:
        Series of ATR values named "atr". First `period` values are NaN.
    """
    prev_close = close.shift(1)
    tr = cast(
        pd.Series,
        pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1),
    )
    alpha = 1.0 / period
    result = cast(
        pd.Series, tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    )
    result.name = "atr"
    return result


def macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Compute MACD line, signal line, and histogram.

    Args:
        prices: Price series with a DatetimeIndex.
        fast: Fast EMA span. Defaults to 12.
        slow: Slow EMA span. Defaults to 26.
        signal: Signal EMA span. Defaults to 9.

    Returns:
        DataFrame with columns ["macd", "signal", "histogram"].
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram},
        index=prices.index,
    )


def bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Compute Bollinger Bands (upper, middle, lower).

    Args:
        prices: Price series with a DatetimeIndex.
        period: Rolling window size. Defaults to 20.
        num_std: Number of standard deviations for band width. Defaults to 2.0.

    Returns:
        DataFrame with columns ["upper", "middle", "lower"].
    """
    middle = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return pd.DataFrame(
        {"upper": upper, "middle": middle, "lower": lower},
        index=prices.index,
    )


def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Compute ratio of current volume to its rolling mean.

    Args:
        volume: Volume series with a DatetimeIndex.
        period: Rolling window size. Defaults to 20.

    Returns:
        Series of volume ratios named "volume_ratio". First `period-1` values are NaN.
    """
    rolling_mean = volume.rolling(period).mean()
    result = volume / rolling_mean
    result.name = "volume_ratio"
    return result
