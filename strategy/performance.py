"""
performance.py — Full performance analytics for the cross-sectional stat-arb system.

METRICS COMPUTED:

  Returns:
    Annualized return, Sharpe, Sortino, max drawdown, drawdown duration, Calmar, win rate

  Risk Decomposition:
    SPY beta + correlation, factor PnL attribution, idiosyncratic PnL fraction

  Turnover & Costs:
    Avg daily turnover, total cost drag (bps/year), gross vs net Sharpe

  Signal Quality:
    IC, IC t-stat, ICIR, signal decay table, long/short attribution, quintile spread

DEFINITIONS:
  Sharpe  = mean(daily_ret) / std(daily_ret) × sqrt(252)  [0% risk-free rate]
  Sortino = mean(daily_ret) / downside_std × sqrt(252)
            where downside_std = std(min(r, 0))  [only penalises negative returns]
  Calmar  = annualized_return / |max_drawdown|
  ICIR    = mean(IC) / std(IC)  [IC information ratio; >0.5 considered tradeable]
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import CONFIG

logger = logging.getLogger(__name__)

TDY = CONFIG["trading_days_per_year"]


class PerformanceAnalytics:
    """
    Compute all performance, risk, and signal-quality metrics.

    Parameters
    ----------
    net_pnl        : pd.Series     Net daily P&L in dollars.
    gross_pnl      : pd.Series     Gross (pre-cost) daily P&L.
    cost_series    : pd.Series     Daily costs.
    positions      : pd.DataFrame  Daily signed dollar positions.
    trades         : pd.DataFrame  Daily dollar trades.
    data           : dict          Pipeline output (for returns, spy_returns).
    factors        : dict          Standardized factor exposures.
    factor_returns : pd.DataFrame  Daily Fama-MacBeth slopes (from signal.py).
    signals        : pd.DataFrame  Alpha signals.
    ic_stats       : dict          IC statistics from SignalGenerator.compute_ic().
    factor_exp_ts  : pd.DataFrame  Daily portfolio factor exposures.
    config         : dict          CONFIG.
    """

    def __init__(
        self,
        net_pnl:        pd.Series,
        gross_pnl:      pd.Series,
        cost_series:    pd.Series,
        positions:      pd.DataFrame,
        trades:         pd.DataFrame,
        data:           dict,
        factors:        dict,
        factor_returns: pd.DataFrame,
        signals:        pd.DataFrame,
        ic_stats:       dict,
        factor_exp_ts:  pd.DataFrame,
        config:         dict = CONFIG,
    ):
        self.net_pnl        = net_pnl
        self.gross_pnl      = gross_pnl
        self.cost_series    = cost_series
        self.positions      = positions
        self.trades         = trades
        self.returns        = data["log_returns"]
        self.spy_returns    = data["spy_returns"]
        self.factors        = factors
        self.factor_returns = factor_returns
        self.signals        = signals
        self.ic_stats       = ic_stats
        self.factor_exp_ts  = factor_exp_ts
        self.cfg            = config
        self.NAV            = config["portfolio_notional"]

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def compute_all(self) -> dict:
        """
        Compute the complete tearsheet metrics for all periods.

        Returns
        -------
        dict with keys:
            "full", "train", "val", "test"  —  each is a sub-dict of metrics
            "signal"                         —  signal quality stats
            "costs"                          —  cost analysis
            "risk"                           —  SPY beta/correlation + factor PnL
            "regimes"                        —  per-regime breakdown
            "warnings"                       —  list of red-flag strings
        """
        logger.info("=== Performance Analytics ===")

        # Compute daily returns on NAV
        net_ret   = self.net_pnl / self.NAV
        gross_ret = self.gross_pnl / self.NAV

        result = {}
        result["full"]  = self._period_metrics(net_ret, label="FULL")
        result["train"] = self._period_metrics(net_ret, label="TRAIN",
                                               start=self.cfg["start_date"],
                                               end=self.cfg["train_end"])
        result["val"]   = self._period_metrics(net_ret, label="VAL",
                                               start=self.cfg["val_start"],
                                               end=self.cfg["val_end"])
        result["test"]  = self._period_metrics(net_ret, label="TEST",
                                               start=self.cfg["test_start"],
                                               end=None)

        result["signal"]  = self._signal_metrics()
        result["costs"]   = self._cost_metrics(net_ret, gross_ret)
        result["risk"]    = self._risk_metrics(net_ret)
        result["regimes"] = self._regime_metrics(net_ret)
        result["warnings"]= self._generate_warnings(result)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # PERIOD-LEVEL METRICS
    # ─────────────────────────────────────────────────────────────────────────

    def _period_metrics(
        self,
        daily_ret: pd.Series,
        label: str,
        start: str = None,
        end:   str = None,
    ) -> dict:
        """
        Compute all return-level metrics for a date range.

        Parameters
        ----------
        daily_ret : pd.Series   Daily returns as fractions of NAV.
        label     : str         Period label for logging.
        start     : str or None ISO date string.
        end       : str or None ISO date string.

        Returns
        -------
        dict   Metrics for the period.
        """
        r = daily_ret.copy()
        if start:
            r = r[r.index >= start]
        if end:
            r = r[r.index <= end]

        r = r.dropna()

        if len(r) < 10:
            return {"label": label, "n_days": len(r), "sharpe": np.nan}

        n_years        = len(r) / TDY
        ann_return     = r.mean() * TDY
        ann_vol        = r.std() * np.sqrt(TDY)
        sharpe         = ann_return / ann_vol if ann_vol > 1e-10 else np.nan
        sortino        = self._sortino(r)
        mdd, mdd_days  = self._max_drawdown(r)
        calmar         = ann_return / abs(mdd) if mdd < 0 else np.nan
        win_rate_day   = (r > 0).mean()

        # Weekly win rate
        weekly_pnl   = r.resample("W").sum()
        win_rate_wk  = (weekly_pnl > 0).mean()

        m = {
            "label":           label,
            "n_days":          len(r),
            "n_years":         round(n_years, 2),
            "total_pnl":       round(r.sum() * self.NAV, 2),
            "ann_return":      round(ann_return, 4),
            "ann_vol":         round(ann_vol, 4),
            "sharpe":          round(sharpe, 3),
            "sortino":         round(sortino, 3),
            "max_drawdown":    round(mdd, 4),
            "mdd_duration":    mdd_days,
            "calmar":          round(calmar, 3) if not np.isnan(calmar) else np.nan,
            "win_rate_daily":  round(win_rate_day, 4),
            "win_rate_weekly": round(win_rate_wk, 4),
        }
        logger.info(
            f"[{label}] Sharpe={sharpe:.2f}  Return={ann_return:.1%}  "
            f"MDD={mdd:.1%}  WinRate={win_rate_day:.1%}"
        )
        return m

    # ─────────────────────────────────────────────────────────────────────────
    # SIGNAL QUALITY
    # ─────────────────────────────────────────────────────────────────────────

    def _signal_metrics(self) -> dict:
        """
        Aggregate signal IC statistics and compute quintile attribution.

        Returns
        -------
        dict   IC statistics, decay table, quintile return spread.
        """
        ic = self.ic_stats

        # Quintile return attribution:
        # How much did the long book (top quintile) vs short book (bottom quintile) return?
        long_positions  = self.positions.clip(lower=0)
        short_positions = self.positions.clip(upper=0)
        simple_ret      = np.exp(self.returns) - 1

        long_ret  = (long_positions.shift(1) * simple_ret).sum(axis=1) / self.NAV * 2
        short_ret = (short_positions.shift(1).abs() * simple_ret).sum(axis=1) / self.NAV * 2

        long_ann  = long_ret.mean() * TDY
        short_ann = short_ret.mean() * TDY
        spread    = long_ann - short_ann   # long minus short annualized

        return {
            "ic_mean":     round(ic.get("ic_mean", np.nan), 5),
            "ic_std":      round(ic.get("ic_std", np.nan), 5),
            "icir":        round(ic.get("icir", np.nan), 3),
            "ic_tstat":    round(ic.get("ic_tstat", np.nan), 3),
            "ic_decay":    ic.get("decay", {}),
            "autocorr":    ic.get("autocorrelation", pd.Series(dtype=float)),
            "long_ann_ret":  round(long_ann, 4),
            "short_ann_ret": round(short_ann, 4),
            "quintile_spread": round(spread, 4),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # RISK DECOMPOSITION
    # ─────────────────────────────────────────────────────────────────────────

    def _risk_metrics(self, net_ret: pd.Series) -> dict:
        """
        Compute SPY beta/correlation and factor PnL attribution.

        SPY beta and correlation measure systematic market exposure.
        A good stat-arb strategy should have both near zero (market-neutral).

        Factor PnL attribution decomposes total P&L into:
          - Factor component:       how much came from systematic risk exposures
          - Idiosyncratic component: what the strategy actually targets

        Parameters
        ----------
        net_ret : pd.Series   Daily returns on NAV.

        Returns
        -------
        dict   Risk statistics.
        """
        # Align SPY and net returns
        spy = self.spy_returns.reindex(net_ret.index).dropna()
        common = net_ret.reindex(spy.index).dropna()
        spy    = spy.reindex(common.index)

        # SPY beta via OLS
        if len(common) > 30:
            cov   = np.cov(common.values, spy.values)
            beta  = cov[0, 1] / cov[1, 1] if cov[1, 1] > 1e-10 else np.nan
            corr  = float(np.corrcoef(common.values, spy.values)[0, 1])
        else:
            beta = corr = np.nan

        # Factor PnL attribution
        # At each date, portfolio factor exposure × factor return = factor PnL
        factor_pnl = {}
        for fname in self.factor_returns.columns:
            if fname not in self.factor_exp_ts.columns:
                continue
            exp = self.factor_exp_ts[fname].reindex(self.net_pnl.index)
            fret = self.factor_returns[fname].reindex(self.net_pnl.index)
            # Factor PnL in dollars = exposure × factor_return × NAV
            factor_pnl[fname] = (exp * fret * self.NAV).fillna(0)

        factor_pnl_df       = pd.DataFrame(factor_pnl)
        total_factor_pnl    = factor_pnl_df.sum(axis=1)
        idiosyncratic_pnl   = self.net_pnl - total_factor_pnl

        total_pnl = self.net_pnl.sum()
        idio_frac = idiosyncratic_pnl.sum() / total_pnl if abs(total_pnl) > 1 else np.nan

        factor_contributions = {
            fname: round(float(factor_pnl_df[fname].sum()), 2)
            for fname in factor_pnl_df.columns
        }

        # Average factor exposures of the book
        avg_exposures = {
            fname: round(float(self.factor_exp_ts[fname].mean()), 4)
            for fname in self.factor_exp_ts.columns
        }

        return {
            "spy_beta":                round(float(beta), 4) if not np.isnan(beta) else np.nan,
            "spy_correlation":         round(float(corr), 4) if not np.isnan(corr) else np.nan,
            "total_factor_pnl":        round(float(total_factor_pnl.sum()), 2),
            "idiosyncratic_pnl":       round(float(idiosyncratic_pnl.sum()), 2),
            "idiosyncratic_fraction":  round(float(idio_frac), 4) if not np.isnan(idio_frac) else np.nan,
            "factor_pnl_breakdown":    factor_contributions,
            "avg_factor_exposures":    avg_exposures,
            "factor_pnl_series":       factor_pnl_df,     # kept for visualizations
            "idio_pnl_series":         idiosyncratic_pnl,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # COST ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────

    def _cost_metrics(
        self, net_ret: pd.Series, gross_ret: pd.Series
    ) -> dict:
        """
        Compare gross vs net performance and compute cost drag.

        Annual cost drag (bps) = (gross_ann_return - net_ann_return) × 10000
        This tells you how much the transaction costs hurt the strategy per year.

        Parameters
        ----------
        net_ret   : pd.Series   Net daily returns.
        gross_ret : pd.Series   Gross (pre-cost) daily returns.

        Returns
        -------
        dict   Cost statistics.
        """
        gross_sharpe = (gross_ret.mean() / gross_ret.std() * np.sqrt(TDY)) \
                       if gross_ret.std() > 1e-10 else np.nan
        net_sharpe   = (net_ret.mean() / net_ret.std() * np.sqrt(TDY)) \
                       if net_ret.std() > 1e-10 else np.nan

        gross_ann = gross_ret.mean() * TDY
        net_ann   = net_ret.mean()   * TDY
        cost_drag_bps = (gross_ann - net_ann) * 10_000

        n_years   = len(net_ret) / TDY
        total_costs_usd = self.cost_series.sum()

        turnover_daily = self.trades.abs().sum(axis=1) / self.NAV
        avg_turnover   = turnover_daily.mean()

        return {
            "gross_sharpe":     round(gross_sharpe, 3),
            "net_sharpe":       round(net_sharpe, 3),
            "gross_ann_return": round(gross_ann, 4),
            "net_ann_return":   round(net_ann, 4),
            "annual_cost_bps":  round(cost_drag_bps, 1),
            "total_costs_usd":  round(total_costs_usd, 2),
            "avg_daily_turnover": round(avg_turnover, 4),
            "turnover_series":  turnover_daily,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # REGIME ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────

    def _regime_metrics(self, net_ret: pd.Series) -> list:
        """
        Compute Sharpe and IC for each market regime.

        Parameters
        ----------
        net_ret : pd.Series   Full-history daily net returns.

        Returns
        -------
        list of dict, one per regime.
        """
        results = []
        for name, (start, end) in self.cfg["regimes"].items():
            r = net_ret.copy()
            if start:
                r = r[r.index >= start]
            if end:
                r = r[r.index <= end]
            r = r.dropna()

            if len(r) < 20:
                results.append({"name": name.strip(), "n_days": len(r), "sharpe": np.nan})
                continue

            ann_vol = r.std() * np.sqrt(TDY)
            sharpe  = r.mean() * TDY / ann_vol if ann_vol > 1e-10 else np.nan

            # IC for this regime
            ic_series = self.ic_stats.get("daily_ic", pd.Series(dtype=float))
            if start:
                ic_series = ic_series[ic_series.index >= start]
            if end:
                ic_series = ic_series[ic_series.index <= end]
            ic_mean = ic_series.mean() if len(ic_series) > 0 else np.nan

            results.append({
                "name":    name.strip(),
                "n_days":  len(r),
                "sharpe":  round(float(sharpe), 3),
                "ic_mean": round(float(ic_mean), 5) if not np.isnan(ic_mean) else np.nan,
                "ann_ret": round(float(r.mean() * TDY), 4),
            })
            logger.info(
                f"[{name.strip()}] Sharpe={sharpe:.2f}  IC={ic_mean:.4f}  N={len(r)}d"
            )

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # WARNINGS
    # ─────────────────────────────────────────────────────────────────────────

    def _generate_warnings(self, result: dict) -> list:
        """
        Scan all metrics and generate human-readable red flags.

        Parameters
        ----------
        result : dict   Output of compute_all() so far (without warnings).

        Returns
        -------
        list[str]   Warning messages; empty if no issues.
        """
        warnings = []
        threshold = self.cfg["oos_degradation_threshold"]

        # OOS degradation: test Sharpe vs train Sharpe
        train_s = result["train"].get("sharpe", np.nan)
        test_s  = result["test"].get("sharpe", np.nan)
        if not (np.isnan(train_s) or np.isnan(test_s)) and train_s > 0:
            ratio = test_s / train_s
            if ratio < threshold:
                warnings.append(
                    f"OOS DEGRADATION: Test Sharpe ({test_s:.2f}) is {ratio:.0%} of "
                    f"Train Sharpe ({train_s:.2f}) — below the {threshold:.0%} threshold. "
                    "Strategy may be overfit to the 2015-2020 period."
                )

        # High SPY beta (should be near zero for market-neutral strategy)
        spy_beta = result["risk"].get("spy_beta", np.nan)
        if not np.isnan(spy_beta) and abs(spy_beta) > 0.2:
            warnings.append(
                f"HIGH SPY BETA: beta = {spy_beta:.3f}. A market-neutral strategy "
                "should have |beta| < 0.1.  Check for factor exposure leakage."
            )

        # Low IC (signal has no predictive power)
        ic_mean = result["signal"].get("ic_mean", np.nan)
        if not np.isnan(ic_mean) and abs(ic_mean) < 0.01:
            warnings.append(
                f"WEAK SIGNAL: IC = {ic_mean:.5f} (< 0.01).  "
                "The idiosyncratic residual signal has very low predictive power."
            )

        # Decaying IC in recent regime
        regimes = result.get("regimes", [])
        if regimes:
            recent = regimes[-1]
            if recent.get("ic_mean", np.nan) is not np.nan:
                if not np.isnan(recent["ic_mean"]) and recent["ic_mean"] < ic_mean * 0.5:
                    warnings.append(
                        f"IC DECAY in recent regime ('{recent['name']}'): "
                        f"IC = {recent['ic_mean']:.5f} vs full-period {ic_mean:.5f}. "
                        "Signal predictive power is degrading."
                    )

        # High turnover → costs eating returns
        avg_to = result["costs"].get("avg_daily_turnover", np.nan)
        if not np.isnan(avg_to) and avg_to > 0.15:
            warnings.append(
                f"HIGH TURNOVER: {avg_to:.1%}/day average.  "
                "Rebalancing is frequent — transaction costs may dominate."
            )

        if not warnings:
            logger.info("No warnings generated — all checks passed.")
        else:
            for w in warnings:
                logger.warning(f"WARNING: {w}")

        return warnings

    # ─────────────────────────────────────────────────────────────────────────
    # METRIC CALCULATION HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _sortino(returns: pd.Series) -> float:
        """
        Compute annualized Sortino ratio.

        Sortino = mean(r) × sqrt(252) / downside_std
        where downside_std = std(min(r, 0)) × sqrt(252).

        Unlike Sharpe, Sortino only penalizes DOWNSIDE volatility.
        An investor who cares only about avoiding losses should prefer Sortino.

        Parameters
        ----------
        returns : pd.Series   Daily returns.

        Returns
        -------
        float   Annualized Sortino ratio.
        """
        downside = returns[returns < 0]
        if len(downside) < 5:
            return np.nan
        down_std = downside.std() * np.sqrt(TDY)
        return returns.mean() * TDY / down_std if down_std > 1e-10 else np.nan

    @staticmethod
    def _max_drawdown(returns: pd.Series) -> tuple:
        """
        Compute maximum drawdown and its duration.

        Max drawdown = largest peak-to-trough decline in cumulative returns.
        Duration = number of days from peak to recovery (or to end of period).

        Parameters
        ----------
        returns : pd.Series   Daily returns.

        Returns
        -------
        (max_drawdown, duration_days) : (float, int)
        """
        cum = returns.cumsum()
        running_max = cum.expanding().max()
        drawdown    = cum - running_max   # always ≤ 0
        mdd         = float(drawdown.min())

        # Compute duration of the maximum drawdown
        is_uw = drawdown < 0
        max_dur = 0
        cur_dur = 0
        for uw in is_uw:
            if uw:
                cur_dur += 1
                max_dur = max(max_dur, cur_dur)
            else:
                cur_dur = 0

        return mdd, max_dur
