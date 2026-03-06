# Step 5 — Triple-Barrier Labeling (COMPLETE)

## Status
- Implementation: `src/trading/labeling/barriers.py`
- Tests: `tests/unit/test_barriers.py` (14 tests, all passing)
- Exported via: `from trading.labeling import triple_barrier_labels`

---

## Scope

One labeling function in `src/trading/labeling/barriers.py`:

| Function | Output |
|---|---|
| `triple_barrier_labels(close, volatility, pt_sl, max_hold)` | `pd.Series` of `{-1, 0, +1}` |

Exported via `src/trading/labeling/__init__.py`.
Tests: `tests/unit/test_barriers.py`.

---

## Function Signature

```python
# src/trading/labeling/barriers.py

def triple_barrier_labels(
    close: pd.Series,
    volatility: pd.Series,
    pt_sl: tuple[float, float] = (1.0, 1.0),
    max_hold: int = 10,
) -> pd.Series: ...
```

---

## Logic

For each bar `i`:

1. If `volatility[i]` is NaN → label is NaN (warm-up, skip).
2. `price = close[i]`, `vol = volatility[i]`
3. `pt, sl = pt_sl`
4. Upper barrier: `price * (1 + pt * vol)` if `pt > 0` else `+inf` (disabled).
5. Lower barrier: `price * (1 - sl * vol)` if `sl > 0` else `-inf` (disabled).
6. Forward window: `close[i+1 : i+max_hold+1]` (truncated at end of series).
7. Scan window bar by bar:
   - First bar where `close[j] >= upper` → label = **+1** (profit-take hit).
   - First bar where `close[j] <= lower` → label = **-1** (stop-loss hit).
   - No hit in window → label = **0** (time barrier).
8. Profit-take takes priority if both hit the same bar (impossible with daily close, but defined for safety).

Return a `pd.Series` named `"label"`, same index as `close`.

---

## Edge Cases

1. **NaN volatility** — label is NaN; non-NaN vol always produces an integer label.
2. **`pt_sl = (0, 0)`** — both barriers disabled; all labels are 0 (time barrier always).
3. **`max_hold = 0`** — empty forward window; all labels are 0.
4. **Near end of series** — window truncates to available bars; still returns 0 (not NaN).
5. **`pt = 0`** — upper barrier disabled, no +1 labels possible.
6. **`sl = 0`** — lower barrier disabled, no -1 labels possible.
7. **Mismatched index** — `volatility` is aligned to `close` via index before use.
8. **Non-positive prices** — raise `ValueError`.

---

## Test Cases (`tests/unit/test_barriers.py`)

### Output metadata
1. `test_output_type` — returns `pd.Series`
2. `test_output_index` — index matches `close`
3. `test_output_name` — name is `"label"`

### Label validity
4. `test_labels_are_valid` — all non-NaN values in `{-1, 0, 1}`
5. `test_nan_where_vol_nan` — NaN label where vol is NaN (warm-up period)

### Barrier logic
6. `test_uptrend_hits_pt` — strong uptrend, tight SL → majority labels +1
7. `test_downtrend_hits_sl` — strong downtrend, tight PT → majority labels -1
8. `test_flat_hits_time` — flat prices → all non-NaN labels are 0
9. `test_pt_zero_no_positive_labels` — `pt=0`: no +1 labels
10. `test_sl_zero_no_negative_labels` — `sl=0`: no -1 labels
11. `test_max_hold_zero` — `max_hold=0`: all non-NaN labels are 0

### Edge cases
12. `test_near_end_no_nan` — last few bars (truncated window) return 0, not NaN
13. `test_non_positive_prices_raises` — raises `ValueError`

### Integration
14. `test_real_data_smoke` — run on real SPY close + ewm_volatility; no errors, correct length
