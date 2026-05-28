# Cross-Sectional Tech Stat-Arb — Five-Factor Mean-Reversion System

A fully self-contained, walk-forward **statistical arbitrage backtester** applied to a universe of **50 US tech stocks**.

Data runs from **2015**; the model is trained on 2015–2020 and backtested on out-of-sample data from **2021 onwards**. The system neutralizes common factor exposure with a five-factor risk model, then generates a mean-reversion alpha signal from idiosyncratic residuals via Fama-MacBeth cross-sectional regression. Dollar-neutral long/short portfolios are sized weekly, with full transaction cost modelling.

---

## Strategy Overview

### 1. Factor Model (`strategy/factor_model.py`)

Five factors are computed daily for each stock in the universe:

| Factor | Construction |
|---|---|
| **Market Beta** | Rolling 252-day OLS β vs SPY |
| **Momentum** | 12-1 month cumulative return (Jegadeesh-Titman, 1993) |
| **Short Reversal** | Trailing 21-day return (negative sign) |
| **Size** | Log market cap, 21-day smoothed |
| **Volatility** | 63-day realized vol (negative sign) |

At each date, all factor exposures are cross-sectionally winsorized (1st/99th pct) and standardized to zero mean / unit variance across the universe.

### 2. Signal Generation (`strategy/signal.py`)

A daily cross-sectional OLS regression strips factor returns from realized returns (Fama-MacBeth 1973):

```
r_i,t = α_t + Σ_k γ_k,t × F_k,i,t + ε_i,t
```

The residual `ε_i,t` is the factor-neutral idiosyncratic return. A 63-day rolling z-score of `ε_i,t` is computed per stock; the **signal is the negative z-score** — betting on mean-reversion of stocks that have run far above/below their recent idiosyncratic history.

### 3. Portfolio Construction (`strategy/portfolio.py`)

- Weekly rebalance (Monday)
- Long top-10 / Short bottom-10 stocks by signal
- Dollar-neutral: long notional ≈ short notional
- 20% single-stock cap
- Turnover buffer: hold positions until rank drops outside top 20

### 4. Transaction Costs (`config.py`)

| Cost Type | Value |
|---|---|
| Per-share commission | $0.005/share each way |
| Market impact | 10 bps of trade value each way |
| Borrow cost | 25 bps/year on shorts |

### 5. Walk-Forward Split

| Period | Dates |
|---|---|
| Training | 2015-01-01 – 2020-12-31 |
| Validation | 2021-01-01 – 2022-12-31 |
| Test (OOS) | 2023-01-01 – present |

Signal parameters are chosen on the training period; validation and test are fully held out.

---

## Backtest Results (Out-of-Sample Test Period)

| Metric | Value |
|---|---|
| Annualized Return (net) | 5.25% |
| Annualized Volatility | 10.74% |
| Sharpe Ratio (net) | 0.49 |
| Sortino Ratio | 0.75 |
| Max Drawdown | -16.57% |
| Market Beta | 0.0003 |
| IC Mean (test period) | 0.021 |
| IC Mean (full sample) | 0.012 |
| IC t-stat | 2.27 |
| Gross Sharpe | 1.00 |
| Daily Turnover | 23.3% |

The strategy is near-market-neutral (β ≈ 0.0003). The gross→net Sharpe compression (1.00 → 0.49) reflects realistic transaction cost drag at ~632 bps/year. The IC improves meaningfully in the test period (0.021 vs 0.012 full-sample), driven by stronger mean-reversion dynamics in the AI boom regime (2023–present).

---

## Repository Structure

```
.
├── main.py                   # Pipeline orchestrator (run this)
├── config.py                 # All parameters — never hardcode
├── requirements.txt
├── strategy/
│   ├── data_pipeline.py      # yfinance download + Parquet caching
│   ├── factor_model.py       # Five-factor cross-sectional risk model
│   ├── signal.py             # Fama-MacBeth regression + z-score signal
│   ├── portfolio.py          # Dollar-neutral portfolio construction
│   ├── backtest.py           # Walk-forward backtesting engine
│   ├── performance.py        # Sharpe, Sortino, IC, drawdown analytics
│   └── visualizations.py     # 11 research charts + summary table
└── results/
    └── summary.py            # Console tearsheet printer
```

---

## Installation & Usage

```bash
pip install -r requirements.txt
python main.py
```

Output artifacts written to `results/`:

```
results/metrics.json               # full metrics dict
results/00_summary_table.png       # colour-coded metrics table
results/01_cumulative_pnl.png
results/02_rolling_sharpe.png
results/03_drawdown.png
results/04_factor_exposure.png
results/05_quintile_spread.png
results/06_ic_over_time.png
results/07_turnover.png
results/08_pnl_attribution.png
results/09_correlation_heatmap.png
results/10_trade_pnl_histogram.png
```

Raw data is downloaded automatically on first run via yfinance and cached as Parquet files.

---

## Universe

50 US tech stocks: NVDA, AAPL, MSFT, GOOGL, META, AVGO, AMD, ORCL, QCOM, TXN, NOW, INTU, AMAT, LRCX, SNPS, CDNS, KLAC, ADI, MRVL, CRM, PANW, CRWD, ADBE, NFLX, SHOP, UBER, INTC, HPQ, DELL, FTNT, NET, DDOG, WDAY, TTD, ZS, TEAM, HUBS, MDB, OKTA, PATH, SNOW, RBLX, DUOL, MNDY, ANSS, PTC, EPAM, CTSH, PSTG, ARM
