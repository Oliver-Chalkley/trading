import numpy as np
import pandas as pd


def ewm_volatility(prices: pd.Series, span: int = 20) -> pd.Series:
    """Compute EWM rolling volatility of log returns.

    Args:
        prices: Price series (e.g. close prices) with a DatetimeIndex.
        span: Span parameter for the EWM standard deviation. Defaults to 20.

    Returns:
        Series of EWM volatility, same index as prices, named "ewm_vol".
        First span-1 values are NaN (min_periods=span).

    Raises:
        ValueError: If any price is <= 0.
    """
    if (prices <= 0).any():
        raise ValueError("All prices must be positive (> 0).")

    log_ret = np.log(prices / prices.shift(1))
    vol = log_ret.ewm(span=span, min_periods=span).std()
    vol.name = "ewm_vol"
    return vol
