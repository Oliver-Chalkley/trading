from typing import Any

import pandas as pd
from statsmodels.tsa.stattools import adfuller


def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for each column of a feature DataFrame.

    Args:
        df: DataFrame of feature values. NaN values are excluded per column.

    Returns:
        DataFrame with rows [mean, std, skew, kurtosis, min, max] and columns
        matching the input.

    Raises:
        ValueError: If `df` is empty.
    """
    if df.empty:
        raise ValueError("df must not be empty")
    stats = {
        "mean": df.mean(),
        "std": df.std(),
        "skew": df.skew(),
        "kurtosis": df.kurt(),
        "min": df.min(),
        "max": df.max(),
    }
    return pd.DataFrame(stats).T


def autocorrelation(series: pd.Series, lags: list[int]) -> pd.Series:
    """Compute Pearson autocorrelation at specified lags.

    Args:
        series: Input time series.
        lags: List of integer lags to compute autocorrelation at.

    Returns:
        Series of autocorrelation values indexed by lag, named "autocorrelation".

    Raises:
        ValueError: If the series has fewer non-NaN observations than max(lags) + 1.
    """
    clean = series.dropna()
    if len(clean) < max(lags) + 1:
        raise ValueError(
            f"Series has {len(clean)} non-NaN observations but needs at least "
            f"{max(lags) + 1} for the requested lags."
        )
    values = [clean.autocorr(lag=lag) for lag in lags]
    return pd.Series(values, index=lags, name="autocorrelation")


def adf_stationarity(series: pd.Series) -> dict[str, Any]:
    """Run the Augmented Dickey-Fuller test for stationarity.

    Args:
        series: Input time series (prices or returns).

    Returns:
        Dict with keys:
            - "stat" (float): ADF test statistic.
            - "p_value" (float): p-value of the test.
            - "is_stationary" (bool): True if p_value < 0.05.

    Raises:
        ValueError: If series has fewer than 20 non-NaN observations.
    """
    clean = series.dropna()
    if len(clean) < 20:  # noqa: PLR2004
        raise ValueError(
            f"Series has {len(clean)} non-NaN observations; ADF requires at least 20."
        )
    stat, p_value, *_ = adfuller(clean)
    return {
        "stat": float(stat),
        "p_value": float(p_value),
        "is_stationary": bool(p_value < 0.05),  # noqa: PLR2004
    }


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the pairwise Pearson correlation matrix.

    Args:
        df: DataFrame of feature values.

    Returns:
        Square DataFrame of Pearson correlations.

    Raises:
        ValueError: If `df` is empty or has fewer than 2 columns.
    """
    if df.empty:
        raise ValueError("df must not be empty")
    if df.shape[1] < 2:  # noqa: PLR2004
        raise ValueError("df must have at least 2 columns for a correlation matrix")
    return df.corr(method="pearson")
