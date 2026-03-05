# Strategic Implementation Plan: MLDP-Style Backtesting Suite

This document outlines the technical architecture and implementation strategy for building an institutional-grade, ML-centric backtesting suite in Python. This plan adheres to the methodologies pioneered by Marcos López de Prado (MLDP), specifically focusing on eliminating backtest overfitting and improving signal-to-noise ratios via Meta-Labeling.

---

## 1. The Philosophical Foundation

Traditional backtesting often fails because it treats historical data as a single path. To follow de Prado's *Advances in Financial Machine Learning* (AFML) style, we must shift from **Backtesting as a Demonstration** to **Backtesting as an Experiment**.

- **Financial Bars over Time Bars**: Markets do not process information at fixed clock intervals.
- **Path Dependency**: Exits should be based on price action (Triple-Barrier), not just a closing price at T+n.
- **Meta-Labeling**: Separating the "Side" (Buy/Sell) from the "Size" (Execute/Ignore).

---

## 2. The Technical Stack (100% Open Source)

| Layer | Component | Selection |
|---|---|---|
| Data Engine | Bar Generation | Pandas + Numpy (Custom Dollar/Volume Bar logic) |
| Feature Engineering | Indicators | Pandas-TA |
| Research & Labeling | MLDP Framework | VectorBT (Core Engine) |
| Machine Learning | Model Training | Scikit-Learn / XGBoost |
| Visualization | TradingView UI | lightweight-charts-python |
| Statistical Validation | Overfitting Prevention | Custom implementation of Deflated Sharpe Ratio |

---

## 3. Phase-by-Phase Execution Plan

### Phase 1: Data Structuring (Information Processing)

Standard OHLC time bars are sampled at fixed intervals, leading to heteroskedasticity. We will implement **Dollar Bars**.

- **Implementation**: Aggregate trades/ticks until a threshold of "Total Dollar Value Exchanged" is met.
- **Result**: Data that is more "normally distributed" and statistically easier for ML models to digest.

### Phase 2: The Triple-Barrier Method (The Labeling Engine)

Standard "Return at T+n" labeling ignores what happens during the trade. We implement the **Triple-Barrier Method**:

- **Horizontal Upper Barrier**: Profit-take (PT).
- **Horizontal Lower Barrier**: Stop-loss (SL).
- **Vertical Barrier**: Time-out (expiration).

The barriers are **dynamic**, scaled by a volatility estimator (e.g., a 20-period Exponentially Weighted Standard Deviation of returns).

### Phase 3: Meta-Labeling (The Machine Learning Filter)

This is the core of MLDP's "Proper" backtesting.

- **The Primary Model**: A simple heuristic (e.g., Trend Following) that generates a "Side" (Long or Short).
- **The Meta-Model**: A binary classifier (Random Forest) that looks at the features (RSI, Volatility, Spread) at the time of the signal and predicts **Binary 1** (Success) or **Binary 0** (Failure).
- **Execution**: We only take the trade if the Meta-Model predicts a success probability > 0.5.

### Phase 4: Purged & Embargoed Cross-Validation

To prevent "look-ahead bias" and leakage, we cannot use standard K-Fold CV.

- **Purging**: Remove observations from the training set that occur immediately after the testing set.
- **Embargoing**: Remove observations at the end of the training set that might be correlated with the testing set.

### Phase 5: Statistical Evaluation

We ignore the nominal Sharpe Ratio. Instead, we calculate the **Deflated Sharpe Ratio (DSR)**:

$$DSR = P\left[\widehat{SR} > SR_0\right]$$

This accounts for the number of trials (N) and the variance of the trials, effectively penalizing you for "cherry-picking" the best-performing parameters.

---

## 4. Visualization Layer (TradingView Style)

To achieve the TradingView look-and-feel within Python, we use **lightweight-charts**.

- **Integration**: Generate a trade list from the VectorBT results.
- **Mapping**: Each trade is mapped to a "Marker" (Green Up Arrow for Long, Red Down Arrow for Short).
- **Interactivity**: The chart remains interactive (zoom/pan) within a browser window or Jupyter cell, allowing for visual verification of the Triple-Barrier exits.

---

## 5. Integration with Claude Code

Claude acts as a Quantitative Researcher. Given raw CSV/API data, Claude executes the following workflow:

1. **Compute Volatility**: Write the EWM volatility script.
2. **Apply Barriers**: Use VectorBT to find where price hits the PT, SL, or Time-out.
3. **Feature Matrix**: Generate technical features for the Meta-Labeling model.
4. **Meta-Train**: Run the Random Forest and output the Classification Report.
5. **Visualize**: Generate the lightweight-charts code to show the "Proprietary View."

---

## Next Steps

Begin with the foundation: **Data Structuring and Labeling** — implementing the Dynamic Volatility calculation and Triple-Barrier Labeling.
