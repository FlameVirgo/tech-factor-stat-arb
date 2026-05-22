"""
signal.py — Idiosyncratic return signal via cross-sectional Fama-MacBeth regression.

SIGNAL CONSTRUCTION (4 steps):

Step 1 — Daily cross-sectional OLS regression at each date t:
    r_i,t  =  alpha_t  +  sum_k(gamma_k,t × F_k,i,t)  +  epsilon_i,t

    Where:
      r_i,t     = stock i's log return on date t
      F_k,i,t   = stock i's standardized exposure to factor k on date t
      gamma_k,t = factor return (cross-sectional regression coefficient)
      epsilon_i,t = idiosyncratic return (residual unexplained by the factors)

    This is the Fama-MacBeth (1973) cross-sectional regression.

Step 2 — Residuals epsilon_i,t are the factor-neutralized idiosyncratic returns.
    A positive residual means the stock outperformed what its factor exposures
    predicted.  A negative residual means it underperformed.

Step 3 — Rolling 63-day z-score of residuals for each stock:
    z_i,t = (epsilon_i,t - mean(epsilon_i, t-63:t)) / std(epsilon_i, t-63:t)

    WHY ROLLING Z-SCORE?
    A single day's residual is noisy.  The rolling z-score captures how
    extreme TODAY's residual is relative to the RECENT HISTORY of residuals
    for that stock.  High positive z = stock has been surprising on the
    upside recently.

Step 4 — Signal = NEGATIVE z-score:
    signal_i,t = -z_i,t

    WHY NEGATIVE?
    Mean reversion: a stock that has had persistently large positive idiosyncratic
    moves is likely to mean-revert (it has "gotten ahead of itself").
    Shorting high positive z → long stocks with large negative z.
    This is a classic stat-arb / mean-reversion signal.

ANTI-LOOKAHEAD ENFORCEMENT:
    - Factor exposures F_k,i,t use only rolling windows ending at t.
    - The cross-sectional regression at date t uses r_i,t and F_k,i,t —
      both known at the close of t.
    - The rolling z-score uses rolling().mean() and rolling().std(), which
      at date t reference only t-63 through t.
    - The SIGNAL at close_t is used to trade at OPEN_{t+1} (enforced in backtest.py).

OUTPUTS:
    generate() returns (signals, factor_returns, residuals)
      signals       : pd.DataFrame (dates × tickers), the alpha signal
      factor_returns: pd.DataFrame (dates × factors), Fama-MacBeth slopes
      residuals     : pd.DataFrame (dates × tickers), raw idiosyncratic returns
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import CONFIG

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Generate the idiosyncratic return mean-reversion signal.

    Parameters
    ----------
    data    : dict   Output of DataPipeline.run().
    factors : dict   Output of FactorModel.compute_all_factors().
    config  : dict   CONFIG from config.py.
    """

    def __init__(self, data: dict, factors: dict, config: dict = CONFIG):
        self.returns  = data["log_returns"]    # (dates, tickers)
        self.factors  = factors                # dict of (dates, tickers) DataFrames
        self.cfg      = config
        self.universe = data["universe"]
        self.factor_names = list(factors.keys())

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def generate(self) -> tuple:
        """
        Run the full signal generation pipeline.

        Returns
        -------
        signals        : pd.DataFrame (dates × tickers)  alpha signals (−z-score)
        factor_returns : pd.DataFrame (dates × factors)   Fama-MacBeth slopes
        residuals      : pd.DataFrame (dates × tickers)   raw idiosyncratic returns
        """
        logger.info("=== Signal Generation ===")

        residuals, factor_returns = self._daily_cross_sectional_regression()
        z_scores = self._rolling_zscore(residuals)
        signals  = -z_scores   # negative: mean-reversion signal

        logger.info(
            f"Signal stats: mean={signals.stack().mean():.4f}, "
            f"std={signals.stack().std():.4f}, "
            f"non-NaN={signals.notna().sum().sum():,}"
        )

        return signals, factor_returns, residuals

    def compute_ic(self, signals: pd.DataFrame, forward_returns: pd.DataFrame) -> dict:
        """
        Compute Information Coefficient at multiple forward horizons.

        IC is the Spearman rank correlation between the signal at time t and the
        forward return over the next N days.  It measures the signal's predictive
        power.

        IC > 0.03 is considered meaningful for a daily equity signal.
        |ICIR| = |mean(IC)| / std(IC) > 0.5 is considered tradeable.

        Parameters
        ----------
        signals         : pd.DataFrame (dates × tickers) alpha signal.
        forward_returns : pd.DataFrame (dates × tickers) log returns.

        Returns
        -------
        dict with keys:
            "daily_ic"        pd.Series    IC at each date (1-day horizon)
            "ic_mean"         float        Mean IC over full period
            "ic_std"          float        Std of IC
            "icir"            float        IC Information Ratio = mean/std
            "ic_tstat"        float        t-statistic of mean IC
            "decay"           dict         IC at each decay horizon (1,5,10,21 days)
            "autocorrelation" pd.Series    Signal autocorrelation at lags 1..10
        """
        horizons  = self.cfg["ic_decay_horizons"]
        decay_ics = {}

        for h in horizons:
            # Forward return = sum of next h daily log returns
            fwd = forward_returns.shift(-h).rolling(h, min_periods=h).sum() if h > 1 \
                  else forward_returns.shift(-1)
            ic_series = self._compute_daily_ic(signals, fwd)
            decay_ics[h] = ic_series.mean()

        # 1-day IC time series (most important for daily strategy)
        fwd_1 = forward_returns.shift(-1)
        daily_ic = self._compute_daily_ic(signals, fwd_1)

        ic_mean = daily_ic.mean()
        ic_std  = daily_ic.std()
        icir    = ic_mean / ic_std if ic_std > 1e-10 else 0.0
        n       = daily_ic.notna().sum()
        ic_tstat = ic_mean / (ic_std / np.sqrt(n)) if n > 0 and ic_std > 1e-10 else 0.0

        # Signal autocorrelation: how persistent is the signal?
        # High autocorrelation → slow-decaying signal → lower turnover needed
        autocorr = pd.Series({
            lag: signals.stack().autocorr(lag=lag) for lag in range(1, 11)
        }, name="autocorrelation")

        return {
            "daily_ic":        daily_ic,
            "ic_mean":         float(ic_mean),
            "ic_std":          float(ic_std),
            "icir":            float(icir),
            "ic_tstat":        float(ic_tstat),
            "decay":           decay_ics,
            "autocorrelation": autocorr,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — CROSS-SECTIONAL OLS REGRESSION
    # ─────────────────────────────────────────────────────────────────────────

    def _daily_cross_sectional_regression(self) -> tuple:
        """
        Run the Fama-MacBeth cross-sectional OLS at every date.

        At each date t:
          X = (n_stocks × n_factors) matrix of factor exposures
          y = (n_stocks,) vector of that day's log returns
          OLS: y = X @ gamma + epsilon
          residuals[t] = epsilon  (idiosyncratic returns)
          factor_returns[t] = gamma  (factor returns / Fama-MacBeth slopes)

        Only stocks with BOTH valid returns AND valid factor exposures on date t
        are included in that day's regression.  Others get NaN residual.

        ANTI-LOOKAHEAD:
          factor exposures at t use only data through t.
          returns r_i,t is the day t return — known at close of t.

        Returns
        -------
        residuals      : pd.DataFrame (dates × tickers)
        factor_returns : pd.DataFrame (dates × factors)
        """
        dates = self.returns.index
        tickers = self.universe
        residuals      = pd.DataFrame(np.nan, index=dates, columns=tickers)
        factor_returns = pd.DataFrame(np.nan, index=dates, columns=self.factor_names)

        logger.info(f"  Running cross-sectional OLS on {len(dates)} dates …")

        for date in dates:
            ret = self.returns.loc[date]   # pd.Series (tickers,)

            # Build factor matrix: (n_tickers × n_factors)
            factor_rows = []
            for fname in self.factor_names:
                if date in self.factors[fname].index:
                    factor_rows.append(self.factors[fname].loc[date])
                else:
                    factor_rows.append(pd.Series(np.nan, index=tickers))
            F = pd.DataFrame(factor_rows, index=self.factor_names).T   # (tickers × factors)

            # Keep only rows where BOTH return and ALL factors are non-NaN
            combined = pd.concat([ret.rename("ret"), F], axis=1)
            valid    = combined.dropna()

            if len(valid) < self.cfg["n_long"] + self.cfg["n_short"] + len(self.factor_names):
                # Too few stocks to run a meaningful regression
                continue

            y = valid["ret"].values
            X = np.column_stack([
                np.ones(len(valid)),                    # intercept
                valid[self.factor_names].values,        # factor exposures
            ])

            # Least-squares fit (equivalent to OLS MLE for Gaussian errors)
            coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            eps = y - X @ coefs   # residuals (idiosyncratic returns)

            residuals.loc[date, valid.index] = eps
            factor_returns.loc[date, self.factor_names] = coefs[1:]   # skip intercept

        return residuals, factor_returns

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — ROLLING Z-SCORE OF RESIDUALS
    # ─────────────────────────────────────────────────────────────────────────

    def _rolling_zscore(self, residuals: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling 63-day z-score of residuals for each stock independently.

        FORMULA:
            z_i,t = (epsilon_i,t - mean(epsilon_i, t-62:t)) / std(epsilon_i, t-62:t)

        The rolling window captures how unusual today's idiosyncratic return is
        relative to the recent distribution for that stock.  This is more robust
        than using the full-history mean/std because it adapts to regime changes.

        ANTI-LOOKAHEAD:
            rolling().mean() and rolling().std() at date t use only data
            from dates t-62 through t (inclusive).  No future information.

        Parameters
        ----------
        residuals : pd.DataFrame   Raw idiosyncratic returns (dates × tickers).

        Returns
        -------
        pd.DataFrame   Rolling z-scores, same shape.
        """
        window      = self.cfg["zscore_window"]
        min_periods = self.cfg["zscore_min_periods"]

        rolling_mean = residuals.rolling(window, min_periods=min_periods).mean()
        rolling_std  = residuals.rolling(window, min_periods=min_periods).std(ddof=1)

        # Divide, clipping std away from zero to prevent numerical blow-up
        z = (residuals - rolling_mean) / rolling_std.clip(lower=1e-8)
        return z

    # ─────────────────────────────────────────────────────────────────────────
    # IC HELPER
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_daily_ic(
        self, signals: pd.DataFrame, forward_ret: pd.DataFrame
    ) -> pd.Series:
        """
        Compute daily cross-sectional Spearman rank IC.

        IC_t = spearman_correlation(signal_i,t, forward_return_i,t+h)
        across all stocks i with valid data on date t.

        Spearman (rank-based) correlation is used rather than Pearson because:
        - It is robust to outlier returns (e.g., earnings surprises).
        - It captures monotonic relationships without requiring linearity.

        Parameters
        ----------
        signals     : pd.DataFrame   Signal values (dates × tickers).
        forward_ret : pd.DataFrame   Forward returns (dates × tickers).

        Returns
        -------
        pd.Series   Daily IC values.
        """
        ic_values = []
        dates     = []

        for date in signals.index:
            s = signals.loc[date].dropna()
            if date not in forward_ret.index:
                continue
            f = forward_ret.loc[date].reindex(s.index).dropna()
            common = s.index.intersection(f.index)
            if len(common) < 10:
                continue

            corr, _ = spearmanr(s.loc[common], f.loc[common])
            ic_values.append(corr)
            dates.append(date)

        return pd.Series(ic_values, index=pd.DatetimeIndex(dates), name="IC")
