"""
factor_model.py — Five-factor cross-sectional risk model for the tech universe.

THE FIVE FACTORS:
    1. MARKET BETA    — rolling 252-day OLS beta vs SPY
    2. MOMENTUM       — 12-1 month cumulative return (Jegadeesh-Titman, 1993)
    3. SHORT REVERSAL — trailing 21-day return with negative sign
    4. SIZE           — log market cap (21-day smoothed)
    5. VOLATILITY     — 63-day realized vol with negative sign

WHY THESE FACTORS?
    They represent well-documented, persistent cross-sectional risk premia and
    are the standard control variables in empirical asset pricing.  By computing
    factor exposures and neutralizing them from our alpha signal, we ensure the
    strategy's edge is genuinely idiosyncratic — not just a disguised beta tilt,
    momentum tilt, etc.

CROSS-SECTIONAL STANDARDIZATION:
    At each date, each factor is standardized across the universe:
      1. Winsorize at 1st/99th percentile (removes extreme outliers)
      2. Demean and divide by std  →  exposure ∈ roughly (−3, +3)
    This makes factor exposures comparable across dates and factors.

ANTI-LOOKAHEAD:
    All rolling computations are strictly backward-looking (rolling().mean(),
    rolling().var(), etc.).  The cross-sectional regression in signal.py also
    uses only data available at time t.

OUTPUTS:
    compute_all_factors() returns dict[str, pd.DataFrame]:
        Each DataFrame has shape (n_dates, n_tickers) with standardized
        cross-sectional factor exposures.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import mstats

from config import CONFIG

logger = logging.getLogger(__name__)


class FactorModel:
    """
    Build and standardize all five risk factors for the cross-sectional universe.

    Parameters
    ----------
    data   : dict   Output of DataPipeline.run() — must contain 'log_returns',
                    'market_caps', 'spy_returns'.
    config : dict   CONFIG from config.py.
    """

    def __init__(self, data: dict, config: dict = CONFIG):
        self.returns    = data["log_returns"]    # (dates, tickers)
        self.mktcaps    = data["market_caps"]    # (dates, tickers)
        self.spy        = data["spy_returns"]    # (dates,)
        self.cfg        = config
        self.universe   = data["universe"]

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def compute_all_factors(self) -> dict:
        """
        Compute all five factors, cross-sectionally standardize each one.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys: "beta", "momentum", "reversal", "size", "vol"
            Each DataFrame: shape (n_dates, n_tickers), standardized exposures.
        """
        logger.info("Computing factor model …")

        raw_factors = {
            "beta":     self._compute_beta(),
            "momentum": self._compute_momentum(),
            "reversal": self._compute_reversal(),
            "size":     self._compute_size(),
            "vol":      self._compute_vol(),
        }

        # Cross-sectional standardize: winsorize then scale to mean=0, std=1
        factors = {
            name: self._cross_sectional_standardize(df, name)
            for name, df in raw_factors.items()
        }

        for name, df in factors.items():
            n_valid = df.notna().sum().sum()
            logger.info(f"  [{name}] {n_valid:,} valid factor observations.")

        return factors

    # ─────────────────────────────────────────────────────────────────────────
    # FACTOR CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_beta(self) -> pd.DataFrame:
        """
        Rolling 252-day OLS market beta for each stock vs SPY.

        FORMULA:
            beta_i,t = Cov(r_i, r_SPY; rolling 252d) / Var(r_SPY; rolling 252d)

        This is the standard CAPM beta.  Stocks with beta > 1 move more than the
        market on average; beta < 1 moves less.

        ANTI-LOOKAHEAD:
            rolling(252).cov() at date t uses only data up to and including date t.

        Returns
        -------
        pd.DataFrame   Raw (un-standardized) betas, shape (dates, tickers).
        """
        logger.info("  Computing MARKET BETA …")
        window      = self.cfg["beta_window"]
        min_periods = self.cfg["beta_min_periods"]

        # Efficient vectorized rolling covariance / variance
        spy_aligned = self.spy.reindex(self.returns.index)
        spy_var = spy_aligned.rolling(window, min_periods=min_periods).var()

        betas = pd.DataFrame(index=self.returns.index, columns=self.universe, dtype=float)
        for ticker in self.universe:
            r = self.returns[ticker]
            cov = r.rolling(window, min_periods=min_periods).cov(spy_aligned)
            betas[ticker] = cov / spy_var.replace(0, np.nan)

        return betas

    def _compute_momentum(self) -> pd.DataFrame:
        """
        Jegadeesh-Titman 12-1 month momentum factor.

        FORMULA:
            mom_i,t = r_i,[t-252 : t-21]   (cumulative log return, skipping last month)

        WHY SKIP THE LAST MONTH?
            Short-term returns exhibit reversal (not momentum).  Including the most
            recent month would contaminate the momentum signal with reversal noise.
            The Jegadeesh-Titman (1993) paper established this 12-1 construction.

        ANTI-LOOKAHEAD:
            log_returns.shift(21) at date t gives returns from t-21 backwards,
            ensuring we never use future data.

        Returns
        -------
        pd.DataFrame   Raw momentum scores, shape (dates, tickers).
        """
        logger.info("  Computing MOMENTUM (12-1 month) …")
        window = self.cfg["momentum_window"]
        skip   = self.cfg["momentum_skip"]

        # Shift by `skip` days: skip.shift(skip) at time t gives r_{t-skip}
        # Then sum over the preceding (window - skip) days
        ret_shifted = self.returns.shift(skip)
        momentum    = ret_shifted.rolling(window - skip, min_periods=(window - skip) // 2).sum()
        return momentum

    def _compute_reversal(self) -> pd.DataFrame:
        """
        Short-term reversal: trailing 21-day return with NEGATIVE sign.

        FORMULA:
            reversal_i,t = −sum(r_i, t-21:t)

        WHY NEGATIVE?
            Stocks that have recently outperformed tend to mean-revert in the
            subsequent month (De Bondt & Thaler, 1985; Jegadeesh, 1990).
            A negative sign makes the factor exposure work the same way as
            the other factors: high exposure = attractive, low exposure = unattractive.

        Returns
        -------
        pd.DataFrame   Raw reversal factor, shape (dates, tickers).
        """
        logger.info("  Computing SHORT-TERM REVERSAL (21-day) …")
        window = self.cfg["reversal_window"]
        trailing_return = self.returns.rolling(window, min_periods=window // 2).sum()
        return -trailing_return   # negative sign: recent gainers are short candidates

    def _compute_size(self) -> pd.DataFrame:
        """
        Size factor: smoothed log market capitalization.

        FORMULA:
            size_i,t = log(mean(mktcap_i, t-21:t))

        WHY SMOOTH?
            Raw market cap fluctuates daily with prices.  A 21-day average
            reduces noise and avoids overfitting to short-term price moves.

        WHY LOG?
            Market caps span several orders of magnitude (e.g., AAPL at $3T vs
            PATH at $5B).  Log compresses this range and makes the distribution
            more symmetric, which is a prerequisite for meaningful standardization.

        Returns
        -------
        pd.DataFrame   Raw log market cap scores, shape (dates, tickers).
        """
        logger.info("  Computing SIZE (log market cap) …")
        window       = self.cfg["size_smooth_window"]
        smoothed_cap = self.mktcaps.rolling(window, min_periods=window // 2).mean()
        return np.log(smoothed_cap.clip(lower=1.0))   # clip to avoid log(0)

    def _compute_vol(self) -> pd.DataFrame:
        """
        Volatility factor: 63-day realized volatility with NEGATIVE sign.

        FORMULA:
            vol_i,t = −std(r_i, t-63:t) × sqrt(252)

        WHY NEGATIVE?
            Higher volatility stocks demand a higher risk premium, making them
            LESS attractive per unit of expected return on a risk-adjusted basis.
            Negative sign: low vol = high factor exposure = attractive.

        WHY 63 DAYS?
            ~3 months is short enough to reflect current market conditions but
            long enough to estimate variance reliably.

        Returns
        -------
        pd.DataFrame   Raw negative volatility, shape (dates, tickers).
        """
        logger.info("  Computing VOLATILITY (63-day realized vol) …")
        window = self.cfg["vol_window"]
        tdy    = self.cfg["trading_days_per_year"]
        realized_vol = self.returns.rolling(window, min_periods=window // 2).std() * np.sqrt(tdy)
        return -realized_vol   # negative sign: high vol = unattractive

    # ─────────────────────────────────────────────────────────────────────────
    # CROSS-SECTIONAL STANDARDIZATION
    # ─────────────────────────────────────────────────────────────────────────

    def _cross_sectional_standardize(
        self, df: pd.DataFrame, factor_name: str
    ) -> pd.DataFrame:
        """
        Apply cross-sectional winsorization then z-scoring at every date.

        At each date t:
          1. Winsorize: clip values below 1st percentile and above 99th percentile.
          2. Z-score:   subtract cross-sectional mean, divide by cross-sectional std.

        WHY CROSS-SECTIONAL (NOT TIME-SERIES)?
            We rank and bet across stocks simultaneously.  We care about relative
            exposures across the universe on a given day, not absolute level.
            Time-series standardization would not respect this.

        Parameters
        ----------
        df          : pd.DataFrame   Raw factor values (dates × tickers).
        factor_name : str            Name for logging.

        Returns
        -------
        pd.DataFrame   Standardized factor exposures, shape (dates, tickers).
        """
        pct = self.cfg["winsorize_pct"]
        result = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

        for date, row in df.iterrows():
            valid = row.dropna()
            if len(valid) < 5:
                continue   # not enough cross-sectional variation

            # Step 1: Winsorize at 1st/99th percentile
            lo, hi = valid.quantile(pct), valid.quantile(1 - pct)
            clipped = valid.clip(lower=lo, upper=hi)

            # Step 2: Z-score (mean=0, std=1 across the universe)
            mu  = clipped.mean()
            sig = clipped.std()
            if sig < 1e-10:
                result.loc[date, valid.index] = 0.0
            else:
                result.loc[date, valid.index] = (clipped - mu) / sig

        return result.astype(float)
