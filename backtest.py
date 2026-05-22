"""
backtest.py — Event-driven backtester for the OU stat-arb system.

LOOP STRUCTURE:
  Burn-in: first ROLLING_WINDOW (252) days skipped — no positions taken.

  Each subsequent day t:
    A. If it is a rebalance day (every REBAL_FREQ=5 trading days):
       1. Fit FactorModel on trailing 252-day log-return window.
       2. Extract residuals E (252, N).
       3. Fit SignalGenerator on E → ou_params, z_scores.
       4. Read today's z-score from the last row of z_scores.
       5. Build new target positions via PortfolioConstructor.
       6. Execute trades (new_pos − old_pos), apply transaction costs.
       7. Store factor model and OU params for intra-week stop-loss checks.

    B. On non-rebalance days:
       1. Call FactorModel.project_new_return() with today's log-returns.
       2. Call SignalGenerator.zscore_single() → today's z-scores.
       3. For any open position where |z_t| ≥ STOP_Z, close it.
          Treat closure as a trade → apply transaction costs.

    C. ALWAYS: carry positions forward, compute daily P&L.
       P&L_t = sum_i(pos_{i,t-1} × r_{i,t} / P_{i,t-1})
             = sum_i(pos_{i,t-1} × log_return_{i,t})   [approximation]

TRANSACTION COSTS:
    cost_per_share  = $0.005 per share traded (each way)
    market_impact   = 10 bps of notional traded (each way)
    borrow_cost     = 25 bps / year on daily average short exposure

ANTI-LOOKAHEAD:
    Positions decided on day t are only applied from day t+1.
    Implemented via positions.shift(1) before computing P&L.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from factors   import FactorModel,    ROLLING_WINDOW
from signals   import SignalGenerator, STOP_Z
from portfolio import PortfolioConstructor

logger = logging.getLogger(__name__)

REBAL_FREQ         =  5      # rebalance every 5 trading days
COST_PER_SHARE     =  0.005  # $ per share each way
MARKET_IMPACT_BPS  = 10      # bps of notional each way
BORROW_COST_BPS    = 25      # bps / year on short exposure (easy-to-borrow tech)
TRADING_DAYS_YEAR  = 252


class Backtester:
    """
    Walk-forward OU stat-arb backtester.

    Parameters
    ----------
    log_returns : pd.DataFrame  (T, N) daily log returns.
    prices      : pd.DataFrame  (T, N) adjusted close prices (for share-count cost calc).
    target_notional : float     Total gross notional ($).
    """

    def __init__(
        self,
        log_returns:      pd.DataFrame,
        prices:           pd.DataFrame,
        target_notional:  float = 1_000_000,
    ):
        self.log_returns = log_returns.copy()
        self.prices      = prices.reindex(columns=log_returns.columns)
        self.tickers     = list(log_returns.columns)
        self.dates       = log_returns.index

        self.factor_model = FactorModel()
        self.signal_gen   = SignalGenerator()
        self.port_ctor    = PortfolioConstructor(target_notional=target_notional)

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Execute the full backtest.

        Returns
        -------
        dict with keys:
            positions   pd.DataFrame  (T, N) signed dollar positions (post-shift).
            pnl_gross   pd.Series     (T,)   daily gross P&L ($).
            pnl_net     pd.Series     (T,)   daily net P&L after costs ($).
            costs       pd.Series     (T,)   total transaction costs ($).
            trades      pd.DataFrame  (T, N) dollar amount traded each day.
            metrics     dict          summary stats (computed by backtest; full analytics in report.py).
        """
        T = len(self.dates)
        N = len(self.tickers)

        # Storage
        positions_raw = pd.DataFrame(0.0, index=self.dates, columns=self.tickers)
        trades        = pd.DataFrame(0.0, index=self.dates, columns=self.tickers)
        costs_series  = pd.Series(0.0, index=self.dates)

        current_pos    = pd.Series(0.0, index=self.tickers)
        last_fit       = None    # FactorModel fit_result
        last_signal    = None    # SignalGenerator result
        running_levels = {}      # {ticker: cumulative spread level since last rebal}
        rebal_counter  = 0

        for idx in range(T):
            date = self.dates[idx]

            # ── Burn-in: skip first ROLLING_WINDOW days ────────────────────
            if idx < ROLLING_WINDOW:
                positions_raw.loc[date] = current_pos
                continue

            # ── Slice the trailing return window ───────────────────────────
            window_returns = self.log_returns.iloc[idx - ROLLING_WINDOW : idx]

            is_rebal = (rebal_counter % REBAL_FREQ == 0)

            if is_rebal:
                # ── A. Rebalance day ───────────────────────────────────────
                last_fit, last_signal, new_pos = self._rebalance(
                    window_returns, date
                )
                # Seed running levels from last fitted cumsum values
                running_levels = {
                    t: p["last_level"]
                    for t, p in last_signal["ou_params"].items()
                    if "last_level" in p
                }
                trade_today = new_pos - current_pos
                cost        = self._transaction_cost(trade_today, date)
                trades.loc[date]     = trade_today
                costs_series[date]   = cost
                current_pos          = new_pos.copy()

            else:
                # ── B. Non-rebalance day: intra-week stop-loss check ───────
                if last_fit is not None and last_signal is not None:
                    r_today = self.log_returns.loc[date]
                    e_today = self.factor_model.project_new_return(r_today, last_fit)
                    z_today = self.signal_gen.zscore_single(
                        e_today, last_signal["ou_params"], running_levels
                    )
                    trade_today, cost = self._apply_stoploss(
                        current_pos, z_today, date
                    )
                    if trade_today.abs().sum() > 0:
                        trades.loc[date]   = trade_today
                        costs_series[date] = cost
                        current_pos        = current_pos + trade_today

            positions_raw.loc[date] = current_pos
            rebal_counter += 1

        # ── Anti-lookahead: apply decisions from next day ──────────────────
        positions = positions_raw.shift(1).fillna(0.0)

        # ── Daily gross P&L ───────────────────────────────────────────────
        # P&L_t = sum_i( pos_{i,t-1} × log_return_{i,t} )
        pnl_gross = (positions * self.log_returns).sum(axis=1)

        # ── Borrow cost on shorts (daily accrual) ──────────────────────────
        short_exposure = positions.clip(upper=0).abs().sum(axis=1)
        borrow_cost    = short_exposure * (BORROW_COST_BPS / 1e4) / TRADING_DAYS_YEAR

        pnl_net = pnl_gross - costs_series - borrow_cost

        return {
            "positions": positions,
            "pnl_gross": pnl_gross,
            "pnl_net":   pnl_net,
            "costs":     costs_series + borrow_cost,
            "trades":    trades,
        }

    # ── Rebalance ──────────────────────────────────────────────────────────

    def _rebalance(
        self,
        window_returns: pd.DataFrame,
        date:           pd.Timestamp,
    ) -> tuple:
        """Fit models and build new target positions for a rebalance day."""
        # 1. Fit factor model
        fit_result = self.factor_model.fit(window_returns)

        # 2. Extract residuals and fit OU signals
        E = fit_result["E"]
        signal_result = self.signal_gen.fit(E)

        # 3. Read today's z-score (last row of the in-sample z_scores)
        z_today = signal_result["z_scores"].iloc[-1]

        # 4. Build positions
        new_pos = self.port_ctor.build_weights(
            z_scores  = z_today,
            ou_params = signal_result["ou_params"],
            B         = fit_result["B"],
            tickers   = fit_result["tickers"],
        )
        new_pos = new_pos.reindex(self.tickers, fill_value=0.0)

        n_tradeable = len(signal_result["tradeable"])
        logger.debug(
            f"[{date.date()}] REBAL  tradeable={n_tradeable}  "
            f"gross=${new_pos.abs().sum():,.0f}"
        )

        return fit_result, signal_result, new_pos

    # ── Stop-Loss ──────────────────────────────────────────────────────────

    def _apply_stoploss(
        self,
        current_pos: pd.Series,
        z_today:     pd.Series,
        date:        pd.Timestamp,
    ) -> Tuple[pd.Series, float]:
        """
        Close any position where |z_t| ≥ STOP_Z.

        Returns (trade_series, total_cost).
        """
        trade = pd.Series(0.0, index=self.tickers)
        for ticker in self.tickers:
            pos = current_pos.get(ticker, 0.0)
            if abs(pos) == 0:
                continue
            z = z_today.get(ticker, np.nan)
            if np.isnan(z):
                continue
            if abs(z) >= STOP_Z:
                trade[ticker] = -pos   # close the position
                logger.debug(
                    f"[{date.date()}] STOP-LOSS {ticker}: z={z:.2f}, pos=${pos:,.0f}"
                )
        cost = self._transaction_cost(trade, date)
        return trade, cost

    # ── Transaction Cost ───────────────────────────────────────────────────

    def _transaction_cost(
        self, trade: pd.Series, date: pd.Timestamp
    ) -> float:
        """
        Compute round-trip transaction cost for a set of trades.

        cost = sum_i(
            cost_per_share × |shares_traded_i|
          + market_impact_bps/1e4 × |dollar_traded_i|
        )
        """
        total = 0.0
        for ticker, dollar_trade in trade.items():
            if abs(dollar_trade) < 1e-6:
                continue
            price = self.prices.loc[date, ticker] if date in self.prices.index else np.nan
            if np.isnan(price) or price <= 0:
                # Fallback: use only market impact if price unavailable
                total += abs(dollar_trade) * MARKET_IMPACT_BPS / 1e4
                continue
            shares = abs(dollar_trade) / price
            total += COST_PER_SHARE * shares
            total += abs(dollar_trade) * MARKET_IMPACT_BPS / 1e4
        return total
