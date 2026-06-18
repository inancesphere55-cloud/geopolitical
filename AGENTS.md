# AGENTS.md — Project Context

## Goal
Build a complete energy & commodity quantitative risk analysis platform (Python backend + live terminal dashboard) for Iran-US conflict scenario modeling.

## Constraints & Preferences
- Backend: yfinance, pandas, numpy, scipy, statsmodels, openpyxl, matplotlib, rich
- Live terminal dashboard: rich TUI with market prices, regime state, VaR/CVaR, scenario outlook
- Single command: double-click `run.bat` (root) or `python backend/terminal_dashboard.py`
- Branding: "SHESHA FINTECH & AI" in the terminal header
- All backend scripts run from `backend/` directory; outputs written to `../data/` and `../outputs/`

## Progress

### Done
- **data_pipeline.py** — downloads Brent, WTI, Gold, VIX (2020–present); 22 features including log returns, 21d/63d realized vol, Brent-WTI spread, Brent/Gold ratio, rolling correlations (Brent-Gold, Brent-VIX), rolling betas, skewness, kurtosis, downside vol; statistical diagnostics (Jarque-Bera normality, ADF stationarity, ann. vol, min/max); feature group summary
- **regime_model.py** — 2-state & 3-state Markov Regression on Brent log returns with lagged VIX exogenous; model comparison via AIC/BIC with automatic degenerate-regime fallback; smoothed crisis probabilities; regime-conditional full correlation matrix (Brent, WTI, Gold, VIX returns per regime); transition matrix & expected regime durations; forward regime probability forecast (5/10/21/30 days); residual diagnostics (Ljung-Box, ARCH, Jarque-Bera); 4-panel chart (prices, probabilities, residuals, squared residuals)
- **scenario_simulator.py** — 30-day Monte Carlo (1,000 paths) across 3 scenarios (Status Quo, Moderate Escalation, Severe Disruption); GARCH(1,1) volatility modeling per regime via MLE; multivariate normal with regime-conditional Cholesky correlations; blended dynamics across scenarios; VaR (90%/95%/99%), CVaR, max drawdown/upside per scenario-asset; full distribution export (simulation_distribution.csv)
- **generate_impact_sheet.py** — 6-sheet Excel workbook: Executive Dashboard (price inputs + sector margins), Detailed Impact Model (cost buildup per sector), Assumptions Register, Sensitivity Analysis (price shock grid 0–70% across sectors), Risk Metrics (VaR/CVaR matrix), Profit Waterfall (Severe Disruption per $100 revenue); conditional formatting (danger/warning/good color fills)
- **generate_report.py** — comprehensive 10-section PDF report with 14 charts (11 2D + 4 3D): price series, return distributions, realized vol, rolling metrics, regime probability, correlation heatmaps, forward forecast, VaR comparison, scenario fan chart, sensitivity heatmap, 3D returns scatter (regime-colored), 3D vol surface, 3D VaR bars, 3D tail risk profile; reportlab-based with embedded figures and styled tables
- **run_all.py** — orchestrates all 5 scripts sequentially; checks output files exist
- **terminal_dashboard.py** — live rich TUI dashboard: runs full pipeline (all 5 scripts), displays prices, regime state, VaR/CVaR, scenario outlook, regime correlations in real-time; `--refresh` flag for continuous refresh every 5 min; shows PDF report status in footer

### In Progress
- (none)

### Blocked
- (none)

### Pipeline Runtime
- Total: ~40s (data_pipeline: 8s, regime_model: 27s, scenario_simulator: 4s, generate_impact_sheet: 2s)

## Key Decisions
- Markov model uses 1-day lagged VIX (shift(1)) as exogenous to avoid lookahead bias
- Regime assignment is dynamic: identifies highest-variance state as crisis
- 3-state model is fitted but if crisis regime has <30 observations, code falls back to 2-state
- Correlation matrices are regularized (1e-6 I) to ensure Cholesky positive definiteness
- GARCH(1,1) estimated per regime via MLE (L-BFGS-B); fallback to simple vol if <30 obs
- All print output uses ASCII characters only (no Unicode) for Windows console compatibility

## Relevant Files
- **backend/data_pipeline.py** — data ingestion & 22-feature engineering pipeline with statistical diagnostics
- **backend/regime_model.py** — Markov regime-switching model with multi-model comparison & residual diagnostics
- **backend/scenario_simulator.py** — GARCH(1,1) multi-asset Monte Carlo with VaR/CVaR risk metrics
- **backend/generate_impact_sheet.py** — 6-sheet Excel financial impact workbook (sensitivity, risk metrics, waterfall)
- **backend/run_all.py** — single-command orchestrator
- **backend/terminal_dashboard.py** — live rich TUI dashboard with pipeline orchestration
- **data/**: CSV outputs (energy_commodity_market_data.csv, market_regime_output.csv, multi_asset_stress_matrix.csv, simulation_distribution.csv, regime_correlations_*.csv)
- **outputs/**: market_regimes.png, Energy_Commodity_Shock_Model.xlsx (6 sheets)
