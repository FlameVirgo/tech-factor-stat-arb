"""
portfolio.py — Cross-sectional ranking portfolio construction.

CONSTRUCTION LOGIC:
    Weekly rebalancing (every Monday):
      1. Rank all stocks by the alpha signal (−z-score of idiosyncratic return).
         Highest signal = most undervalued idiosyncratically = long candidate.
      2. Long the top-N stocks (top quintile, N=10).
         Short the bottom-N stocks (bottom quintile, N=10).
      3. Equal weight within each quintile: $5,000 per stock (on a $100k NAV).
      4. Dollar neutral: total long = total short = $50,000.
      5. Turnover filter: only rebalance a position if its signal rank
         has changed by more than CONFIG["turnover_rank_threshold"] = 10.
         Keeps costs down by avoiding trivial rebalances.

FACTOR EXPOSURE CHECK:
    After constructing each portfolio, compute:
        portfolio_factor_exposure_k = sum_i(w_i / NAV × F_k,i)
    If |exposure| > CONFIG["max_factor_exposure"] = 0.20 for any factor k,
    log a warning.  This flags unintended beta tilts (e.g., if the signal
    accidentally picks mostly high-beta stocks).

OUTPUTS:
    build_positions() returns (positions, trades, factor_exposures_ts):
      positions           pd.DataFrame (dates × tickers)  signed dollar positions
      trades              pd.DataFrame (dates × tickers)  dollar amount traded each day
      factor_exposures_ts pd.DataFrame (dates × factors)  portfolio's factor exposure over time
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import CONFIG

logger = logging.getLogger(__name__)


class PortfolioConstructor:
    """
    Build dollar-neutral long/short portfolios from the alpha signal.

    Parameters
    ----------
    signals : pd.DataFrame   Alpha signal, shape (dates, tickers).
    factors : dict           Standardized factor exposures (from FactorModel).
    prices  : pd.DataFrame   Adjusted close prices (dates × tickers).
    config  : dict           CONFIG from config.py.
    """

    def __init__(
        self,
        signals: pd.DataFrame,
        factors: dict,
        prices:  pd.DataFrame,
        config:  dict = CONFIG,
    ):
        self.signals  = signals
        self.factors  = factors
        self.prices   = prices
        self.cfg      = config
        self.universe = list(signals.columns)

        self._notional_per_side = self.cfg["portfolio_notional"] / 2   # $50k long, $50k short
        self._n_long  = self.cfg["n_long"]
        self._n_short = self.cfg["n_short"]
        self._rank_threshold = self.cfg["turnover_rank_threshold"]

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def build_positions(self) -> tuple:
        """
        Run the full portfolio construction loop across all rebalance dates.

        Returns
        -------
        positions           : pd.DataFrame (dates × tickers), signed dollar positions.
                              Positive = long, Negative = short, 0 = no position.
        trades              : pd.DataFrame (dates × tickers), dollar amount traded each day
                              (positive = bought, negative = sold/shorted).
        factor_exposures_ts : pd.DataFrame (dates × factors), portfolio factor exposure.
        """
        dates      = self.signals.index
        all_tickers = self.universe
        n_factors   = list(self.factors.keys())

        # Identify rebalance dates: every Monday that is also a trading day
        rebal_dates = self._get_rebalance_dates(dates)
        logger.info(
            f"Portfolio construction: {len(rebal_dates)} rebalance dates "
            f"(every Monday from {rebal_dates[0].date()} to {rebal_dates[-1].date()})"
        )

        # Storage
        positions           = pd.DataFrame(0.0, index=dates, columns=all_tickers)
        trades              = pd.DataFrame(0.0, index=dates, columns=all_tickers)
        factor_exp_ts       = pd.DataFrame(np.nan, index=dates, columns=n_factors)

        current_positions   = pd.Series(0.0, index=all_tickers)  # today's live positions
        prev_ranks          = pd.Series(dtype=float)              # ranks from last rebalance

        for i, date in enumerate(dates):
            if date in rebal_dates:
                # ── REBALANCE: compute new target positions ────────────────
                sig_today = self.signals.loc[date].dropna()
                new_positions, new_ranks = self._construct_one_rebalance(
                    signal    = sig_today,
                    prev_pos  = current_positions,
                    prev_ranks= prev_ranks,
                )

                trade_today = new_positions - current_positions
                trades.loc[date]    = trade_today
                current_positions   = new_positions.reindex(all_tickers, fill_value=0.0)
                prev_ranks          = new_ranks

                # ── Factor exposure check ──────────────────────────────────
                self._check_factor_exposures(current_positions, date)

            positions.loc[date] = current_positions

            # Compute rolling portfolio factor exposure (for PnL attribution later)
            factor_exp_ts.loc[date] = self._portfolio_factor_exposure(
                current_positions, date
            )

        return positions, trades, factor_exp_ts

    def compute_turnover(self, trades: pd.DataFrame) -> pd.Series:
        """
        Compute daily portfolio turnover as a percentage of NAV.

        Turnover_t = sum(|trades_i,t|) / NAV × 100%

        High turnover increases transaction costs and reduces net returns.
        A daily turnover of 5-10% is typical for a weekly-rebalance strategy.

        Parameters
        ----------
        trades : pd.DataFrame   Dollar amounts traded each day (dates × tickers).

        Returns
        -------
        pd.Series   Daily turnover as a fraction of NAV.
        """
        return trades.abs().sum(axis=1) / self.cfg["portfolio_notional"]

    # ─────────────────────────────────────────────────────────────────────────
    # CORE CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────────────

    def _construct_one_rebalance(
        self,
        signal:     pd.Series,
        prev_pos:   pd.Series,
        prev_ranks: pd.Series,
    ) -> tuple:
        """
        Build one week's target portfolio from the current signal.

        Logic:
          1. Rank stocks by signal (ascending rank, highest signal = rank N).
          2. Top N = long, bottom N = short.
          3. Apply turnover filter: if rank change ≤ threshold, keep old position.
          4. Scale to dollar-neutral ($50k long, $50k short).
          5. Enforce max 20% single-stock position limit.

        Parameters
        ----------
        signal     : pd.Series   Signal values for all stocks with valid signals.
        prev_pos   : pd.Series   Previous week's dollar positions.
        prev_ranks : pd.Series   Previous week's signal ranks.

        Returns
        -------
        (new_positions, new_ranks)
            new_positions : pd.Series   Signed dollar positions (all tickers).
            new_ranks     : pd.Series   Signal ranks for this week.
        """
        n = len(signal)
        if n < self._n_long + self._n_short + 1:
            # Not enough stocks to form a portfolio
            logger.warning(f"Only {n} valid signals — cannot construct portfolio.")
            return pd.Series(0.0, index=self.universe), pd.Series(dtype=float)

        # Rank (1 = lowest signal, N = highest signal)
        new_ranks = signal.rank(ascending=True, method="average")

        # ── Entry thresholds: strict top-N / bottom-N ────────────────────
        long_candidates  = new_ranks[new_ranks >= new_ranks.nlargest(self._n_long).min()].index
        short_candidates = new_ranks[new_ranks <= new_ranks.nsmallest(self._n_short).max()].index

        # ── Turnover filter: asymmetric entry vs exit thresholds ──────────
        # Entry: must be in top N (strict).
        # Exit:  only exit if rank falls BELOW top (N + threshold).
        #
        # WHY ASYMMETRIC?
        # The old filter checked |rank_change| ≤ threshold, but that fails
        # for the most common case: a stock ranked #10 slips to #11.
        # Its new_tgt becomes 0 (outside top-10), so the sign check fires
        # and the filter never keeps it — even with a very large threshold.
        # The correct fix is a buffer zone: exit threshold > entry threshold.
        # A stock entered at rank #10 only exits when rank drops below #20
        # (with threshold=10). This cuts half the boundary churn.
        buf = self._rank_threshold
        n_total = len(new_ranks)
        long_exit_min  = new_ranks.nlargest(self._n_long + buf).min() \
                         if n_total > self._n_long + buf else new_ranks.min()
        short_exit_max = new_ranks.nsmallest(self._n_short + buf).max() \
                         if n_total > self._n_short + buf else new_ranks.max()

        # Start with entry-only targets (new entrants, equal-weight per side)
        target = pd.Series(0.0, index=signal.index)
        n_long_actual  = len(long_candidates)
        n_short_actual = len(short_candidates)
        if n_long_actual > 0:
            target[long_candidates]  = self._notional_per_side / n_long_actual
        if n_short_actual > 0:
            target[short_candidates] = -self._notional_per_side / n_short_actual

        # Override with kept positions from last week that are still in buffer zone
        if len(prev_ranks) > 0:
            for ticker in prev_pos.index:
                old_pos = prev_pos.get(ticker, 0.0)
                if abs(old_pos) == 0 or ticker not in new_ranks.index:
                    continue
                curr_rank = new_ranks[ticker]
                if old_pos > 0 and curr_rank >= long_exit_min:
                    # Was long, still within the exit buffer → keep at old size
                    target[ticker] = old_pos
                elif old_pos < 0 and curr_rank <= short_exit_max:
                    # Was short, still within the exit buffer → keep at old size
                    target[ticker] = old_pos

        # ── Position limit: no single stock > 20% of NAV ──────────────────
        max_pos = self.cfg["max_single_stock_pct"] * self.cfg["portfolio_notional"]
        target  = target.clip(lower=-max_pos, upper=max_pos)

        # Reindex to full universe (stocks not in portfolio = 0)
        new_positions = target.reindex(self.universe, fill_value=0.0)
        return new_positions, new_ranks

    # ─────────────────────────────────────────────────────────────────────────
    # FACTOR EXPOSURE CHECKS
    # ─────────────────────────────────────────────────────────────────────────

    def _portfolio_factor_exposure(
        self, positions: pd.Series, date: pd.Timestamp
    ) -> pd.Series:
        """
        Compute the portfolio's net exposure to each risk factor.

        portfolio_exposure_k = sum_i(w_i × F_k,i,t)
        where w_i = position_i / NAV  (fraction of portfolio in stock i)

        If this is near zero for all factors, the portfolio is approximately
        factor-neutral — its returns will be driven by idiosyncratic, not
        systematic, risk.

        Parameters
        ----------
        positions : pd.Series   Signed dollar positions.
        date      : pd.Timestamp

        Returns
        -------
        pd.Series   Net exposure to each factor.
        """
        nav     = self.cfg["portfolio_notional"]
        weights = positions / nav   # fractional weights

        exposures = {}
        for fname, factor_df in self.factors.items():
            if date not in factor_df.index:
                exposures[fname] = np.nan
                continue
            f = factor_df.loc[date].reindex(positions.index, fill_value=np.nan)
            exposures[fname] = float((weights * f).sum())

        return pd.Series(exposures)

    def _check_factor_exposures(
        self, positions: pd.Series, date: pd.Timestamp
    ) -> None:
        """
        Log a warning if any factor exposure exceeds CONFIG["max_factor_exposure"].

        A large positive beta exposure means the portfolio is inadvertently
        long the market — it would lose money in a market downturn for reasons
        unrelated to the alpha signal.

        Parameters
        ----------
        positions : pd.Series   Signed dollar positions.
        date      : pd.Timestamp
        """
        exps = self._portfolio_factor_exposure(positions, date)
        threshold = self.cfg["max_factor_exposure"]
        for fname, exp in exps.items():
            if not np.isnan(exp) and abs(exp) > threshold:
                logger.warning(
                    f"[{date.date()}] Factor exposure '{fname}' = {exp:.3f} "
                    f"exceeds threshold {threshold:.2f}."
                )

    # ─────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """
        Return all trading days that fall on a Monday (weekday == 0).

        If the market is closed on a specific Monday (rare), the next available
        trading day within that week is used instead.  In practice, most Mondays
        are trading days.

        Parameters
        ----------
        dates : pd.DatetimeIndex   All available trading days.

        Returns
        -------
        pd.DatetimeIndex   Subset of dates that are Mondays (or first trading day of week).
        """
        # dayofweek == 0 is Monday
        is_monday = dates.dayofweek == self.cfg["rebalance_weekday"]
        return dates[is_monday]
