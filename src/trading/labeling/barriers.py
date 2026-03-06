import numpy as np
import pandas as pd


def triple_barrier_labels(
    close: pd.Series,
    volatility: pd.Series,
    pt_sl: tuple[float, float] = (1.0, 1.0),
    max_hold: int = 10,
) -> pd.Series:
    """Label each bar using the Triple-Barrier method.

    For each bar, three barriers are defined using the EWM volatility:
    - Profit-take (upper): ``price * (1 + pt * vol)``
    - Stop-loss (lower): ``price * (1 - sl * vol)``
    - Vertical (time): ``max_hold`` bars forward

    The label is +1 if the profit-take barrier is hit first, -1 if the
    stop-loss is hit first, and 0 if the vertical barrier is hit (neither
    price barrier touched within the holding window).

    Args:
        close: Close price series with a DatetimeIndex.
        volatility: EWM volatility series aligned to ``close`` (e.g. from
            ``ewm_volatility``). Bars where volatility is NaN are labelled NaN.
        pt_sl: ``(profit_take_multiplier, stop_loss_multiplier)``. Set either
            component to 0 to disable that barrier.
        max_hold: Maximum holding period in bars. ``0`` disables the forward
            window entirely (all non-NaN labels become 0).

    Returns:
        Series of integer labels ``{-1, 0, +1}`` named ``"label"``, same index
        as ``close``. NaN where ``volatility`` is NaN.

    Raises:
        ValueError: If any value in ``close`` is non-positive.
    """
    if (close <= 0).any():
        raise ValueError("close prices must be positive")

    vol_aligned = volatility.reindex(close.index)
    pt, sl = pt_sl
    prices = close.to_numpy()
    vols = vol_aligned.to_numpy()
    n = len(prices)
    labels: list[float] = []

    for i in range(n):
        v = vols[i]
        if np.isnan(v):
            labels.append(np.nan)
            continue

        price = prices[i]
        upper = price * (1.0 + pt * v) if pt > 0 else np.inf
        lower = price * (1.0 - sl * v) if sl > 0 else -np.inf

        end = min(i + max_hold, n - 1)
        label = 0
        for j in range(i + 1, end + 1):
            p_j = prices[j]
            if p_j >= upper:
                label = 1
                break
            if p_j <= lower:
                label = -1
                break

        labels.append(float(label))

    result = pd.Series(labels, index=close.index, name="label")
    # Restore integer dtype where values are not NaN
    return result
