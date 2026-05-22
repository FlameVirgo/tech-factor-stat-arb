"""
backtest.py — Event-driven backtester for the cross-sectional tech stat-arb strategy.

EXECUTION MODEL:
    Signal computed at CLOSE of day t  →  position change at OPEN of day t+1.
    In practice, we approximate "execute at open_{t+1}" using close-to-close
    returns with a 1-day shift on positions — a standard and conservative
    approximation for a daily backtest.

    Specifically: positions.shift(1) means "the position held during day t
    was decided at the close of day t-1."  This correctly prevents any
    same-day trading.

TRANSACTION COST MODEL:
    Each time the position in a stock changes:
      1. Per-share cost:   $0.005 × |shares_traded|
                           where shares_traded = |dollar_change| / price
      2. Market impact:    10bps × |dollar_change_i|
                           (proportional to trade size; conservative for liquid tech)
    Short positions additionally incur:
      3. Borrow cost:      25bps/year × |short_dollar_position_i| / 252 per day

WALK-FORWARD PERIODS:
    train:      2015-2020
    validation: 2021-2022
    test:       2023-present
    Metrics are reported separately for each period.  The test period
    is truly out-of-sample — no parameters were ever chosen based on it.

OUTPUTS:
    run() returns (daily_pnl, gross_pnl, cost_series, book_stats)
      daily_pnl   : pd.Series  net daily P&L in dollars (after all costs)
      gross_pnl   : pd.Series  pre-cost daily P&L in dollars
      cost_series : pd.Series  total costs paid each day
      book_stats  : pd.DataFrame  daily book statistics (gross/net exposure, turnover, etc.)
"""

import logging

import numpy as np
import pandas as pd

from config import CONFIG

logger = logging.getLogger(__name__)


class Backtester:
    """
    Simulate the strategy day by day and compute P&L net of realistic costs.

    Parameters
    ----------
    positions : pd.DataFrame   Signed dollar positions (dates × tickers).
    trades    : pd.DataFrame   Dollar amounts traded on each rebalance day.
    data      : dict           Output of DataPipeline.run().
    config    : dict           CONFIG from config.py.
    """

    def __init__(
        self,
        positions: pd.DataFrame,
        trades:    pd.DataFrame,
        data:      dict,
        config:    dict = CONFIG,
    ):
        self.positions = positions
        self.trades    = trades
        self.prices    = data["prices"]
        self.log_ret   = data["log_returns"]
        self.volumes   = data["volumes"]
        self.cfg       = config

        # Pre-compute simple returns for P&L (log returns approximate simple returns
        # for small daily moves, but exact calculation uses price ratios)
        self.simple_ret = np.exp(self.log_ret) - 1

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> tuple:
        """
        Execute the full backtest simulation.

        Returns
        -------
        daily_pnl   : pd.Series     Net P&L per day in dollars.
        gross_pnl   : pd.Series     Pre-cost P&L per day.
        cost_series : pd.Series     Transaction costs + borrow costs per day.
        book_stats  : pd.DataFrame  Daily book statistics.
        """
        logger.info("=== Backtester ===")

        # ── 1. Gross P&L ─────────────────────────────────────────────────
        # positions.shift(1) ensures we use the position DECIDED at yesterday's
        # close to compute today's P&L.  This is the anti-lookahead lag.
        #
        # ANTI-LOOKAHEAD:  pos_{t-1} × return_t  →  position held during day t,
        # decided at close of day t-1.  No future information used.
        positions_lagged = self.positions.shift(1)
        gross_pnl = (positions_lagged * self.simple_ret).sum(axis=1)
        gross_pnl.name = "gross_pnl"

        # ── 2. Transaction costs (on rebalance days when trades occur) ────
        tc = self._transaction_costs()

        # ── 3. Short borrow costs (accrue daily on short positions) ───────
        borrow = self._borrow_costs(positions_lagged)

        cost_series = (tc + borrow).rename("total_costs")
        daily_pnl   = (gross_pnl - cost_series).rename("net_pnl")

        # ── 4. Book statistics ────────────────────────────────────────────
        book_stats = self._compute_book_stats(positions_lagged, daily_pnl, gross_pnl, tc)

        logger.info(
            f"Backtest complete: total net P&L = ${daily_pnl.sum():,.0f}  |  "
            f"total cost drag = ${cost_series.sum():,.0f}"
        )

        return daily_pnl, gross_pnl, cost_series, book_stats

    def period_mask(self, period: str) -> pd.Series:
        """
        Return a boolean mask for a named walk-forward period.

        Parameters
        ----------
        period : str   One of "train", "val", "test".

        Returns
        -------
        pd.Series   Boolean mask aligned to daily_pnl index.
        """
        periods = {
            "train": (self.cfg["start_date"],    self.cfg["train_end"]),
            "val":   (self.cfg["val_start"],     self.cfg["val_end"]),
            "test":  (self.cfg["test_start"],    str(self.positions.index[-1].date())),
        }
        if period not in periods:
            raise ValueError(f"Unknown period: {period}. Choose from {list(periods)}")

        start, end = periods[period]
        idx = self.positions.index
        return (idx >= start) & (idx <= end)

    # ─────────────────────────────────────────────────────────────────────────
    # COST MODELS
    # ─────────────────────────────────────────────────────────────────────────

    def _transaction_costs(self) -> pd.Series:
        """
        Compute execution costs on days when positions change (rebalance days).

        Cost per trade:
          1. Per-share fee:   $0.005 × |shares|  where shares = |dollar_change| / price
          2. Market impact:   10bps × |dollar_change|

        Both costs are one-way — the total round-trip cost for a position opened
        today and closed next week is 2× this (once at entry, once at exit).

        Parameters
        ----------
        (uses self.trades and self.prices)

        Returns
        -------
        pd.Series   Daily transaction cost in dollars (zero on non-rebalance days).
        """
        cost_per_share  = self.cfg["cost_per_share"]
        impact_bps      = self.cfg["market_impact_bps"] / 10_000

        daily_cost = pd.Series(0.0, index=self.trades.index)

        for date in self.trades.index:
            trade = self.trades.loc[date]
            if trade.abs().sum() == 0:
                continue

            prices_today = self.prices.loc[date].reindex(trade.index)

            # |shares_traded| = |dollar_change| / price
            dollar_change = trade.abs()
            shares_traded = (dollar_change / prices_today.replace(0, np.nan)).fillna(0)

            per_share_cost = (shares_traded * cost_per_share).sum()
            impact_cost    = (dollar_change * impact_bps).sum()

            daily_cost[date] = per_share_cost + impact_cost

        return daily_cost

    def _borrow_costs(self, positions_lagged: pd.DataFrame) -> pd.Series:
        """
        Compute daily short borrow cost on all short positions.

        Borrow cost:  (25bps / year) / 252 days  × |short_dollar_position| per day.

        For easy-to-borrow tech stocks, 25bps/year is the standard "GC rate"
        (general collateral rate).  Hard-to-borrow names can cost 100-500bps/year,
        but our universe consists of liquid, large-cap tech stocks.

        Parameters
        ----------
        positions_lagged : pd.DataFrame   Positions held during each day.

        Returns
        -------
        pd.Series   Daily borrow cost in dollars.
        """
        borrow_daily_rate = self.cfg["borrow_cost_bps"] / 10_000 / self.cfg["trading_days_per_year"]
        # Short positions have negative dollar values
        short_exposure = positions_lagged.clip(upper=0).abs()   # absolute value of shorts
        daily_borrow   = short_exposure.sum(axis=1) * borrow_daily_rate
        return daily_borrow

    # ─────────────────────────────────────────────────────────────────────────
    # BOOK STATISTICS
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_book_stats(
        self,
        positions_lagged: pd.DataFrame,
        net_pnl:          pd.Series,
        gross_pnl:        pd.Series,
        tc:               pd.Series,
    ) -> pd.DataFrame:
        """
        Compute daily book-level statistics for risk monitoring and reporting.

        Statistics computed:
          - gross_long_exposure:   sum of positive positions
          - gross_short_exposure:  sum of absolute value of negative positions
          - net_exposure:          gross_long - gross_short (target ≈ 0)
          - n_longs:               count of long positions
          - n_shorts:              count of short positions
          - gross_exposure:        gross_long + gross_short (total leverage)
          - cumulative_net_pnl:    running sum of net P&L
          - transaction_costs:     daily TC
          - turnover:              |trades| / NAV

        Parameters
        ----------
        positions_lagged : pd.DataFrame
        net_pnl          : pd.Series
        gross_pnl        : pd.Series
        tc               : pd.Series

        Returns
        -------
        pd.DataFrame   Daily book statistics.
        """
        longs  = positions_lagged.clip(lower=0)
        shorts = positions_lagged.clip(upper=0)

        stats = pd.DataFrame({
            "gross_long_exp":  longs.sum(axis=1),
            "gross_short_exp": shorts.abs().sum(axis=1),
            "net_exposure":    positions_lagged.sum(axis=1),
            "gross_exposure":  positions_lagged.abs().sum(axis=1),
            "n_longs":         (longs > 0).sum(axis=1),
            "n_shorts":        (shorts < 0).sum(axis=1),
            "gross_pnl":       gross_pnl,
            "net_pnl":         net_pnl,
            "transaction_costs": tc,
            "cumulative_net_pnl": net_pnl.cumsum(),
        })
        return stats
