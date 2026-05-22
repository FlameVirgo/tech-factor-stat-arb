"""
config.py — Single source of truth for all parameters in the cross-sectional
            tech stat-arb research system.

NEVER hardcode a number in any analysis file.  All tunable quantities live here.
Change one value and the entire pipeline picks it up automatically.
"""

CONFIG = {
    # ─────────────────────────────────────────────────────────────────────────
    # UNIVERSE & BENCHMARK
    # ─────────────────────────────────────────────────────────────────────────
    "universe": [
        "NVDA", "AAPL", "MSFT", "GOOGL", "META", "AVGO", "AMD", "ORCL",
        "QCOM", "TXN", "NOW", "INTU", "AMAT", "LRCX", "SNPS", "CDNS",
        "KLAC", "ADI", "MRVL", "CRM", "PANW", "CRWD", "ADBE", "NFLX",
        "SHOP", "UBER", "INTC", "HPQ", "DELL", "FTNT", "NET", "DDOG",
        "WDAY", "TTD", "ZS", "TEAM", "HUBS", "MDB", "OKTA", "PATH",
        "SNOW", "RBLX", "DUOL", "MNDY", "ANSS", "PTC", "EPAM", "CTSH",
        "PSTG", "ARM",
    ],
    "benchmark": "SPY",

    # ─────────────────────────────────────────────────────────────────────────
    # DATA
    # ─────────────────────────────────────────────────────────────────────────
    "start_date": "2015-01-01",
    "end_date":   None,            # None → today
    "raw_data_dir":       "data/raw/",
    "processed_data_dir": "data/processed/",
    "factors_dir":        "data/factors/",
    "results_dir":        "results/",
    "plots_dir":          "results/plots/",
    "logs_dir":           "results/logs/",
    "tearsheets_dir":     "results/tearsheets/",

    # Data-quality thresholds
    "max_missing_date_pct": 0.10,   # drop date if >10% of stocks are missing
    "max_missing_stock_pct": 0.05,  # drop stock if >5% missing in any rolling year
    "max_fwd_fill_days":    2,      # forward-fill gaps no longer than this many days
    "cache_staleness_days": 1,      # re-download if cache is older than this

    # ─────────────────────────────────────────────────────────────────────────
    # WALK-FORWARD PERIODS
    # Signal parameters are chosen on train; validation and test are truly held out.
    # ─────────────────────────────────────────────────────────────────────────
    "train_end":   "2020-12-31",
    "val_start":   "2021-01-01",
    "val_end":     "2022-12-31",
    "test_start":  "2023-01-01",

    # ─────────────────────────────────────────────────────────────────────────
    # FACTOR MODEL
    # ─────────────────────────────────────────────────────────────────────────
    # All windows are in trading days unless stated otherwise.
    "beta_window":       252,   # rolling OLS beta vs SPY (1 year)
    "beta_min_periods":  126,   # need at least 6 months of data for beta

    # Momentum: 12-month return, skip last month (Jegadeesh-Titman)
    "momentum_window":   252,
    "momentum_skip":     21,    # skip last 21 days to avoid reversal contamination

    # Short-term reversal
    "reversal_window":   21,    # 1-month trailing return (negative sign)

    # Size: log(market cap) smoothed with 21-day average
    "size_smooth_window": 21,

    # Volatility: 63-day realized vol (negative sign: high vol is unattractive)
    "vol_window":        63,

    # Cross-sectional standardization
    "winsorize_pct":     0.01,  # clip at 1st / 99th percentile before standardizing

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL GENERATION
    # ─────────────────────────────────────────────────────────────────────────
    "zscore_window":     63,    # lookback for rolling z-score of factor residuals
    "zscore_min_periods": 21,   # minimum observations needed for a valid z-score

    # ─────────────────────────────────────────────────────────────────────────
    # PORTFOLIO CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────────────
    "rebalance_weekday":  0,    # 0 = Monday (pandas convention)
    "n_long":            10,    # top-N stocks = long quintile
    "n_short":           10,    # bottom-N stocks = short quintile
    "portfolio_notional": 100_000,   # total NAV in USD
    "max_single_stock_pct": 0.20,    # no more than 20% of NAV in one name
    "turnover_rank_threshold": 10,   # exit buffer: keep position until rank drops below top (N+10)

    # Factor exposure limits: warn if |net factor exposure| > threshold
    "max_factor_exposure": 0.20,

    # ─────────────────────────────────────────────────────────────────────────
    # TRANSACTION COSTS
    # ─────────────────────────────────────────────────────────────────────────
    "cost_per_share":       0.005,  # $0.005 per share each way (exchange + SEC fees)
    "market_impact_bps":    10,     # 10bps of trade value each way (market impact)
    "borrow_cost_bps":      25,     # 25bps/year on short positions (easy-to-borrow tech)

    # ─────────────────────────────────────────────────────────────────────────
    # PERFORMANCE ANALYTICS
    # ─────────────────────────────────────────────────────────────────────────
    "trading_days_per_year": 252,
    "risk_free_rate":        0.00,  # annualized; 0% = pure Sharpe

    # IC (information coefficient) analysis
    "ic_rolling_window":  21,       # rolling IC window in days
    "ic_decay_horizons":  [1, 5, 10, 21],  # forward-return horizons for IC decay

    # OOS degradation warning threshold: warn if test Sharpe < train Sharpe × this
    "oos_degradation_threshold": 0.50,

    # ─────────────────────────────────────────────────────────────────────────
    # REGIME LABELS (for regime analysis)
    # ─────────────────────────────────────────────────────────────────────────
    "regimes": {
        "Pre-COVID bull (2015-2019)": ("2015-01-01", "2019-12-31"),
        "COVID crash/recovery (2020)": ("2020-01-01", "2020-12-31"),
        "Rate-hike/tech drawdown (2021-2022)": ("2021-01-01", "2022-12-31"),
        "AI boom (2023-present)": ("2023-01-01", None),
    },
}
