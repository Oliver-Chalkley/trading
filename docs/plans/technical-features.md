# Step 3 — Technical Features (COMPLETE)

## Status
- Implementation: `src/trading/features/indicators.py`
- Tests: `tests/unit/test_indicators.py` (19 tests, all passing)
- Exported via: `from trading.features import rsi, atr, macd, bollinger_bands, volume_ratio`

---

## Scope
Five indicator functions in `src/trading/features/indicators.py`:

| Function | Output |
|---|---|
| `rsi(prices, period=14)` | `pd.Series` named `"rsi"` |
| `atr(high, low, close, period=14)` | `pd.Series` named `"atr"` |
| `macd(prices, fast=12, slow=26, signal=9)` | `pd.DataFrame` cols `["macd","signal","histogram"]` |
| `bollinger_bands(prices, period=20, num_std=2.0)` | `pd.DataFrame` cols `["upper","middle","lower"]` |
| `volume_ratio(volume, period=20)` | `pd.Series` named `"volume_ratio"` |

Exported via `src/trading/features/__init__.py`.
Tests: `tests/unit/test_indicators.py`.

---

## Function Signatures

```python
# src/trading/features/indicators.py

def rsi(prices: pd.Series, period: int = 14) -> pd.Series: ...
def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series: ...
def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame: ...
def bollinger_bands(prices: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame: ...
def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series: ...
```

---

## Logic

### RSI
1. Compute daily price changes: `delta = prices.diff()`
2. Separate gains (`delta.clip(lower=0)`) and losses (`(-delta).clip(lower=0)`)
3. Smooth with Wilder's EWM: `alpha = 1/period`, `adjust=False`
4. `rs = avg_gain / avg_loss`; `rsi = 100 - 100 / (1 + rs)`
5. First `period` values NaN (use `min_periods=period`)

### ATR
1. Compute true range components each bar:
   - `hl = high - low`
   - `hc = (high - close.shift(1)).abs()`
   - `lc = (low - close.shift(1)).abs()`
2. `tr = max(hl, hc, lc)` element-wise via `pd.concat([hl, hc, lc], axis=1).max(axis=1)`
3. Smooth with Wilder's EWM: `alpha = 1/period`, `adjust=False`, `min_periods=period`

### MACD
1. `ema_fast = prices.ewm(span=fast, adjust=False).mean()`
2. `ema_slow = prices.ewm(span=slow, adjust=False).mean()`
3. `macd_line = ema_fast - ema_slow`
4. `signal_line = macd_line.ewm(span=signal, adjust=False).mean()`
5. `histogram = macd_line - signal_line`

### Bollinger Bands
1. `middle = prices.rolling(period).mean()`
2. `std = prices.rolling(period).std()`
3. `upper = middle + num_std * std`
4. `lower = middle - num_std * std`

### Volume Ratio
1. `rolling_mean = volume.rolling(period).mean()`
2. `volume_ratio = volume / rolling_mean`

---

## Edge Cases

1. **Single-row / short input** — output NaN for all values requiring warm-up.
2. **Constant prices (RSI)** — zero gains and zero losses; RS is 0/0 → RSI should be 50 or NaN (handle `avg_loss == 0` case).
3. **Non-positive prices** — `rsi`, `macd`, `bollinger_bands` should raise `ValueError` for non-positive prices.
4. **Bollinger num_std=0** — upper == middle == lower (zero-width bands); valid.
5. **Volume = 0** — `volume_ratio` outputs `NaN` where rolling mean is 0.

---

## Test Cases (`tests/unit/test_indicators.py`)

### RSI
1. `test_rsi_type` — returns `pd.Series`
2. `test_rsi_name` — name is `"rsi"`
3. `test_rsi_range` — all non-NaN values in [0, 100]
4. `test_rsi_leading_nans` — first `period` values are NaN
5. `test_rsi_constant_prices` — constant prices → RSI is NaN (0/0) or 50

### ATR
6. `test_atr_type` — returns `pd.Series`
7. `test_atr_name` — name is `"atr"`
8. `test_atr_non_negative` — all non-NaN values >= 0
9. `test_atr_leading_nans` — first `period` values are NaN

### MACD
10. `test_macd_type` — returns `pd.DataFrame`
11. `test_macd_columns` — columns are `["macd", "signal", "histogram"]`
12. `test_macd_histogram` — histogram == macd - signal for all rows

### Bollinger Bands
13. `test_bb_type` — returns `pd.DataFrame`
14. `test_bb_columns` — columns are `["upper", "middle", "lower"]`
15. `test_bb_ordering` — upper >= middle >= lower for all non-NaN rows

### Volume Ratio
16. `test_vr_type` — returns `pd.Series`
17. `test_vr_name` — name is `"volume_ratio"`
18. `test_vr_constant_volume` — constant volume → ratio is 1.0 after warm-up

### Integration
19. `test_real_data_smoke` — run all five on real SPY data; no errors, correct length
