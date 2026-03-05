# Step 2 — EWM Volatility Estimator (COMPLETE)

## Status
- Implementation: `src/trading/features/volatility.py`
- Tests: `tests/unit/test_volatility.py` (11 tests, all passing)
- Exported via: `from trading.features import ewm_volatility`

---

## Function Signature

```python
# src/trading/features/volatility.py

def ewm_volatility(prices: pd.Series, span: int = 20) -> pd.Series:
    """Compute EWM rolling volatility of log returns.

    Args:
        prices: Price series (e.g. close prices) with a DatetimeIndex.
        span: Span parameter for the EWM standard deviation. Defaults to 20.

    Returns:
        Series of annualised EWM volatility, same index as prices.
        First value is NaN (no prior return available).
    """
```

## Logic
1. Compute log returns: `log_ret = np.log(prices / prices.shift(1))`
2. Apply EWM std: `vol = log_ret.ewm(span=span, min_periods=span).std()`
3. Return `vol` with the same index and name `"ewm_vol"`

## Edge Cases
1. **Constant prices** — all log returns are 0 → volatility is 0 (or NaN if fewer than `span` periods).
2. **Single-row input** — only one price → entire output is NaN.
3. **NaN prices in input** — NaN prices propagate to NaN returns and NaN vol for those bars.
4. **Non-positive prices** — `log(0)` or `log(-1)` is undefined; function should raise `ValueError`.
5. **span=1** — degenerate EWM; each vol equals the absolute log return at that bar.

## Test Cases
1. `test_output_type` — returns `pd.Series`
2. `test_output_index` — output index equals input index
3. `test_output_name` — series name is `"ewm_vol"`
4. `test_leading_nans` — first `span - 1` values are NaN (min_periods=span)
5. `test_non_negative` — all non-NaN values >= 0
6. `test_constant_prices` — flat price series → vol is 0 after warm-up
7. `test_single_row` — single price → output is entirely NaN
8. `test_non_positive_prices_raises` — prices <= 0 raise `ValueError`
9. `test_custom_span` — span=5 warms up faster than span=40
10. `test_real_data` — smoke test on `load_ohlcv()` close prices: no error, correct length
