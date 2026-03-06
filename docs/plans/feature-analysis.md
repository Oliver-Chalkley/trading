# Step 4 — Feature Analysis (COMPLETE)

## Status
- Implementation: `src/trading/features/analysis.py`
- Tests: `tests/unit/test_feature_analysis.py` (20 tests, all passing)
- Exported via: `from trading.features import feature_summary, autocorrelation, adf_stationarity, correlation_matrix`
- Dependency added: `statsmodels`

---

## Scope

Four analysis functions in `src/trading/features/analysis.py`:

| Function | Output |
|---|---|
| `feature_summary(df)` | `pd.DataFrame` — rows are stats, columns are features |
| `autocorrelation(series, lags)` | `pd.Series` — indexed by lag |
| `adf_stationarity(series)` | `dict` — ADF stat, p-value, is_stationary flag |
| `correlation_matrix(df)` | `pd.DataFrame` — square Pearson correlation matrix |

Exported via `src/trading/features/__init__.py`.
Tests: `tests/unit/test_feature_analysis.py`.
New dependency: `statsmodels` (for ADF).

---

## Function Signatures

```python
# src/trading/features/analysis.py

def feature_summary(df: pd.DataFrame) -> pd.DataFrame: ...
def autocorrelation(series: pd.Series, lags: list[int]) -> pd.Series: ...
def adf_stationarity(series: pd.Series) -> dict[str, float | bool]: ...
def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame: ...
```

---

## Logic

### feature_summary
1. Drop NaN rows from each column independently.
2. Compute per-column: `mean`, `std`, `skew`, `kurtosis`, `min`, `max`.
3. Return a DataFrame with those 6 stats as rows, feature columns as columns.
4. Raise `ValueError` if `df` is empty.

### autocorrelation
1. Drop leading NaN from `series`.
2. For each lag in `lags`, compute `series.autocorr(lag=lag)`.
3. Return `pd.Series` indexed by `lags`, named `"autocorrelation"`.
4. Raise `ValueError` if `series` has fewer than `max(lags) + 1` non-NaN observations.

### adf_stationarity
1. Drop NaN from `series`.
2. Run `statsmodels.tsa.stattools.adfuller` with default settings.
3. Return `{"stat": float, "p_value": float, "is_stationary": bool}`.
   - `is_stationary = p_value < 0.05`.
4. Raise `ValueError` if series has fewer than 20 non-NaN observations (ADF minimum).

### correlation_matrix
1. Compute `df.corr(method="pearson")`.
2. Return the resulting square DataFrame.
3. Raise `ValueError` if `df` is empty or has fewer than 2 columns.

---

## Edge Cases

1. **Empty DataFrame** — `feature_summary` and `correlation_matrix` raise `ValueError`.
2. **Single column** — `correlation_matrix` raises `ValueError` (correlation requires ≥2 features).
3. **Short series (ADF)** — `adf_stationarity` raises `ValueError` for < 20 non-NaN obs.
4. **All-NaN column** — `feature_summary` returns NaN for stats of that column (no error).
5. **Constant series (autocorrelation)** — `series.autocorr()` returns NaN; allowed.

---

## Test Cases (`tests/unit/test_feature_analysis.py`)

### feature_summary
1. `test_summary_type` — returns `pd.DataFrame`
2. `test_summary_rows` — rows are `["mean", "std", "skew", "kurtosis", "min", "max"]`
3. `test_summary_columns` — columns match input df columns
4. `test_summary_mean_correct` — mean row matches `df.mean()`
5. `test_summary_empty_raises` — empty df raises `ValueError`

### autocorrelation
6. `test_ac_type` — returns `pd.Series`
7. `test_ac_name` — name is `"autocorrelation"`
8. `test_ac_index` — index equals input `lags`
9. `test_ac_lag0_is_one` — lag-0 autocorrelation is 1.0
10. `test_ac_short_series_raises` — raises `ValueError` when series too short

### adf_stationarity
11. `test_adf_keys` — dict has keys `stat`, `p_value`, `is_stationary`
12. `test_adf_stationary_series` — white noise series is stationary (p < 0.05)
13. `test_adf_nonstationary_series` — random walk is non-stationary (p > 0.05)
14. `test_adf_too_short_raises` — series with < 20 obs raises `ValueError`

### correlation_matrix
15. `test_corr_type` — returns `pd.DataFrame`
16. `test_corr_diagonal` — diagonal is all 1.0
17. `test_corr_symmetric` — matrix equals its transpose
18. `test_corr_single_column_raises` — single-column df raises `ValueError`
19. `test_corr_empty_raises` — empty df raises `ValueError`

### Integration
20. `test_real_data_smoke` — run all four on real SPY features; no errors
