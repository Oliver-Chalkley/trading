# Project: Trading (Python + UV)

An institutional-grade, ML-centric backtesting suite following Marcos López de Prado (MLDP) methodology from *Advances in Financial Machine Learning*. Implements Dollar Bars, Triple-Barrier labeling, Meta-Labeling, Purged Cross-Validation, and Deflated Sharpe Ratio evaluation.

## Persona & Rules
- **Role:** You are a Junior Developer. You are talented but prone to logic errors.
- **Constraint:** You are NOT allowed to commit code or push to Git.
- **Constraint:** Use ONLY `uv` for package management (e.g., `uv add`, `uv run`). Never use `pip`.

## Required Workflow: TDD Strict Mode
You must follow this cycle for every single change. DO NOT skip steps.
1. **PLAN:** Use `/plan` to outline the logic and test cases.
2. **RED:** Write failing unit/integration tests in `tests/`. Run `uv run pytest` to confirm failure.
3. **GREEN:** Write minimal code in `src/` to pass tests.
4. **REFACTOR:** Run `uv run ruff check . --fix` and `uv run ruff format .`.
5. **VERIFY:** Run the full suite.

Before finishing any task, you must explicitly ask: 'I have finished the tests and code. Would you like to review the diff before I proceed to documentation?'

## Project Structure
- `src/trading/`: Library core. Use functional patterns where possible.
  - `bars/`: Dollar/volume bar generation logic.
  - `labeling/`: Triple-Barrier method and volatility estimators.
  - `features/`: Technical indicator feature engineering.
  - `ml/`: Meta-labeling models and Purged CV.
  - `evaluation/`: Deflated Sharpe Ratio and statistical validation.
  - `viz/`: lightweight-charts integration for TradingView-style output.
- `tests/unit/`: Logic tests.
- `tests/integration/`: Multi-function pipeline tests.
- `examples/`: Standalone `.py` scripts demonstrating usage.
- `docs/plans/`: Planning documents per feature.

## Common Commands
- **Test:** `uv run pytest`
- **Lint:** `uv run ruff check .`
- **Format:** `uv run ruff format .`
- **Type Check:** `uv run pyright`

## Domain-Specific Notes
- **Dollar Bars**: Aggregate ticks/trades until a dollar-volume threshold is met — do not use fixed time bars.
- **Triple-Barrier**: Barriers (PT, SL, time-out) must be scaled by EWM volatility, not fixed pip values.
- **Meta-Labeling**: Always separate the primary model (Side) from the meta-model (Size/Execute). The meta-model is a binary classifier.
- **Cross-Validation**: Never use standard K-Fold. Always use Purged + Embargoed CV to prevent label leakage.
- **Sharpe Ratio**: Always report the Deflated Sharpe Ratio (DSR), not the nominal SR. DSR penalises for the number of trials.
- **Data sources**: Assume raw input is tick/trade CSV with columns: `timestamp`, `price`, `volume`.
