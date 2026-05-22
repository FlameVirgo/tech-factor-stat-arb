"""
report.py — Performance analytics, charts, and tearsheet for the OU stat-arb backtest.

OUTPUTS (written to results/):
  metrics.json                  — full dict of all metrics
  00_equity_curve.png           — cumulative net P&L vs gross P&L
  01_drawdown.png               — underwater equity curve
  02_monthly_returns_heatmap.png— calendar heatmap of monthly returns
  03_rolling_sharpe.png         — 63-day rolling Sharpe ratio
  04_cost_breakdown.png         — cumulative costs vs gross P&L
  05_positions_heatmap.png      — signed dollar positions over time (top 20 tickers)
  tearsheet.txt                 — plain-text tearsheet printed to stdout and saved
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import TwoSlopeNorm

logger = logging.getLogger(__name__)

RESULTS_DIR      = Path("results")
TRADING_DAYS_YR  = 252
RISK_FREE_RATE   = 0.0   # annualised


class Reporter:
    """
    Compute performance metrics and generate charts from backtest output.

    Parameters
    ----------
    pnl_gross : pd.Series   Daily gross P&L ($).
    pnl_net   : pd.Series   Daily net P&L ($).
    costs     : pd.Series   Daily total cost ($).
    trades    : pd.DataFrame (dates × tickers) dollar amount traded.
    positions : pd.DataFrame (dates × tickers) signed dollar positions.
    notional  : float       Target gross notional ($).
    """

    def __init__(
        self,
        pnl_gross: pd.Series,
        pnl_net:   pd.Series,
        costs:     pd.Series,
        trades:    pd.DataFrame,
        positions: pd.DataFrame,
        notional:  float = 1_000_000,
    ):
        self.pnl_gross = pnl_gross
        self.pnl_net   = pnl_net
        self.costs     = costs
        self.trades    = trades
        self.positions = positions
        self.notional  = notional

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Compute all metrics, write all charts, print tearsheet."""
        metrics = self._compute_metrics()
        self._save_metrics(metrics)
        self._chart_equity_curve()
        self._chart_drawdown()
        self._chart_monthly_heatmap()
        self._chart_rolling_sharpe()
        self._chart_cost_breakdown()
        self._chart_positions_heatmap()
        self._print_tearsheet(metrics)
        return metrics

    # ── Metrics ────────────────────────────────────────────────────────────

    def _compute_metrics(self) -> dict:
        net   = self.pnl_net.dropna()
        gross = self.pnl_gross.dropna()

        cum_net_pct   = (net   / self.notional).cumsum()
        cum_gross_pct = (gross / self.notional).cumsum()

        ann_ret_net   = float(net.mean()   * TRADING_DAYS_YR / self.notional)
        ann_ret_gross = float(gross.mean() * TRADING_DAYS_YR / self.notional)

        ann_vol_net   = float(net.std()   * np.sqrt(TRADING_DAYS_YR) / self.notional)
        ann_vol_gross = float(gross.std() * np.sqrt(TRADING_DAYS_YR) / self.notional)

        sharpe_net   = self._sharpe(net   / self.notional)
        sharpe_gross = self._sharpe(gross / self.notional)
        sortino_net  = self._sortino(net  / self.notional)

        mdd = self._max_drawdown(cum_net_pct)
        calmar = ann_ret_net / abs(mdd) if abs(mdd) > 1e-6 else np.nan

        total_cost = float(self.costs.sum())
        cost_bps_yr = total_cost / len(net) * TRADING_DAYS_YR / self.notional * 1e4

        total_traded = float(self.trades.abs().sum().sum())
        avg_daily_turnover = float(self.trades.abs().sum(axis=1).mean() / self.notional)
        avg_gross_exp = float(self.positions.abs().sum(axis=1).mean())

        n_trading_days = int((net != 0).sum())
        win_rate = float((net > 0).mean())

        date_start = str(net.index[0].date())
        date_end   = str(net.index[-1].date())
        years      = len(net) / TRADING_DAYS_YR

        return {
            "date_start":       date_start,
            "date_end":         date_end,
            "years":            round(years, 2),
            "ann_return_net":   round(ann_ret_net, 4),
            "ann_return_gross": round(ann_ret_gross, 4),
            "ann_vol_net":      round(ann_vol_net, 4),
            "ann_vol_gross":    round(ann_vol_gross, 4),
            "sharpe_net":       round(sharpe_net, 3),
            "sharpe_gross":     round(sharpe_gross, 3),
            "sortino_net":      round(sortino_net, 3),
            "max_drawdown":     round(float(mdd), 4),
            "calmar_ratio":     round(calmar, 3) if not np.isnan(calmar) else None,
            "total_pnl_gross":  round(float(gross.sum()), 2),
            "total_pnl_net":    round(float(net.sum()), 2),
            "total_cost":       round(total_cost, 2),
            "cost_bps_per_yr":  round(cost_bps_yr, 1),
            "avg_daily_turnover": round(avg_daily_turnover, 4),
            "avg_gross_exposure": round(avg_gross_exp, 2),
            "total_traded":     round(total_traded, 2),
            "win_rate":         round(win_rate, 4),
            "n_trading_days":   n_trading_days,
        }

    def _sharpe(self, returns: pd.Series) -> float:
        mu  = returns.mean() * TRADING_DAYS_YR - RISK_FREE_RATE
        vol = returns.std()  * np.sqrt(TRADING_DAYS_YR)
        return float(mu / vol) if vol > 1e-10 else 0.0

    def _sortino(self, returns: pd.Series) -> float:
        mu      = returns.mean() * TRADING_DAYS_YR - RISK_FREE_RATE
        neg     = returns[returns < 0]
        downvol = neg.std() * np.sqrt(TRADING_DAYS_YR) if len(neg) > 1 else 1e-10
        return float(mu / downvol) if downvol > 1e-10 else 0.0

    def _max_drawdown(self, cum_returns: pd.Series) -> float:
        peak    = cum_returns.cummax()
        dd      = cum_returns - peak
        return float(dd.min())

    # ── Charts ─────────────────────────────────────────────────────────────

    def _chart_equity_curve(self):
        cum_net   = (self.pnl_net   / self.notional).cumsum() * 100
        cum_gross = (self.pnl_gross / self.notional).cumsum() * 100

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(cum_gross.index, cum_gross.values, label="Gross P&L", linewidth=1, alpha=0.7, color="steelblue")
        ax.plot(cum_net.index,   cum_net.values,   label="Net P&L",   linewidth=1.5, color="navy")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title("Cumulative P&L (% of notional)", fontweight="bold")
        ax.set_ylabel("%")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "00_equity_curve.png", dpi=150)
        plt.close(fig)

    def _chart_drawdown(self):
        cum_net = (self.pnl_net / self.notional).cumsum()
        dd      = (cum_net - cum_net.cummax()) * 100

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.fill_between(dd.index, dd.values, 0, alpha=0.4, color="crimson", label="Drawdown")
        ax.plot(dd.index, dd.values, color="crimson", linewidth=0.8)
        ax.set_title("Drawdown (% of notional)", fontweight="bold")
        ax.set_ylabel("%")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "01_drawdown.png", dpi=150)
        plt.close(fig)

    def _chart_monthly_heatmap(self):
        monthly = (self.pnl_net / self.notional * 100).resample("ME").sum()
        monthly.index = monthly.index.to_period("M")
        df = monthly.to_frame("ret")
        df["year"]  = df.index.year
        df["month"] = df.index.month
        pivot = df.pivot(index="year", columns="month", values="ret")
        pivot.columns = [
            "Jan","Feb","Mar","Apr","May","Jun",
            "Jul","Aug","Sep","Oct","Nov","Dec"
        ][:len(pivot.columns)]

        vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 1e-6)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.6)))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", norm=norm)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=7,
                            color="white" if abs(v) > vmax * 0.5 else "black")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Monthly return %")
        ax.set_title("Monthly Returns Heatmap (% of notional)", fontweight="bold")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "02_monthly_returns_heatmap.png", dpi=150)
        plt.close(fig)

    def _chart_rolling_sharpe(self):
        daily_ret = self.pnl_net / self.notional
        roll_sr   = daily_ret.rolling(63).apply(
            lambda x: (x.mean() * TRADING_DAYS_YR) / (x.std() * np.sqrt(TRADING_DAYS_YR))
            if x.std() > 1e-10 else 0.0,
            raw=True
        )
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(roll_sr.index, roll_sr.values, linewidth=1, color="darkorange", label="63-day rolling Sharpe")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.axhline(1, color="green",  linewidth=0.8, linestyle=":", alpha=0.7, label="Sharpe=1")
        ax.set_title("63-Day Rolling Sharpe Ratio", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "03_rolling_sharpe.png", dpi=150)
        plt.close(fig)

    def _chart_cost_breakdown(self):
        cum_gross = self.pnl_gross.cumsum()
        cum_cost  = self.costs.cumsum()
        cum_net   = self.pnl_net.cumsum()

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(cum_gross.index, cum_gross.values / 1e3, label="Gross P&L", color="steelblue", linewidth=1.2)
        ax.plot(cum_net.index,   cum_net.values   / 1e3, label="Net P&L",   color="navy",      linewidth=1.2)
        ax.fill_between(
            cum_cost.index,
            -(cum_cost.values / 1e3),
            0,
            alpha=0.35, color="crimson", label="Cumulative Costs"
        )
        ax.set_title("Gross P&L vs Transaction Costs ($k)", fontweight="bold")
        ax.set_ylabel("$k")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "04_cost_breakdown.png", dpi=150)
        plt.close(fig)

    def _chart_positions_heatmap(self):
        # Show top 20 most actively traded tickers
        active = self.positions.abs().sum().nlargest(20).index
        pos_sub = self.positions[active] / 1e3  # convert to $k

        fig, ax = plt.subplots(figsize=(16, 6))
        vmax = max(pos_sub.abs().values.max(), 1e-3)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(
            pos_sub.T.values, aspect="auto", cmap="RdYlGn", norm=norm,
            interpolation="nearest"
        )
        ax.set_yticks(range(len(active)))
        ax.set_yticklabels(active, fontsize=8)
        # Sparse x-axis dates
        n = len(pos_sub)
        step = max(n // 10, 1)
        ax.set_xticks(range(0, n, step))
        ax.set_xticklabels(
            [pos_sub.index[i].strftime("%Y-%m") for i in range(0, n, step)],
            rotation=45, fontsize=7
        )
        plt.colorbar(im, ax=ax, shrink=0.8, label="Position ($k)")
        ax.set_title("Signed Positions Over Time — Top 20 Tickers ($k)", fontweight="bold")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "05_positions_heatmap.png", dpi=150)
        plt.close(fig)

    # ── Tearsheet ──────────────────────────────────────────────────────────

    def _print_tearsheet(self, m: dict):
        lines = [
            "",
            "=" * 62,
            "  OU STAT-ARB (PCA-LW) — BACKTEST TEARSHEET",
            "=" * 62,
            f"  Period          : {m['date_start']}  →  {m['date_end']}  ({m['years']:.1f} yr)",
            f"  Universe        : 50 US tech stocks",
            f"  Notional        : ${self.notional:,.0f}",
            "",
            "  ─── RETURNS ────────────────────────────────────────────",
            f"  Ann. Return (net)  : {m['ann_return_net']:+.2%}",
            f"  Ann. Return (gross): {m['ann_return_gross']:+.2%}",
            f"  Ann. Volatility    : {m['ann_vol_net']:.2%}",
            "",
            "  ─── RISK-ADJUSTED ──────────────────────────────────────",
            f"  Sharpe (net)    : {m['sharpe_net']:.3f}",
            f"  Sharpe (gross)  : {m['sharpe_gross']:.3f}",
            f"  Sortino (net)   : {m['sortino_net']:.3f}",
            f"  Max Drawdown    : {m['max_drawdown']:.2%}",
            f"  Calmar Ratio    : {m['calmar_ratio']:.3f}" if m['calmar_ratio'] is not None else "  Calmar Ratio    : N/A",
            "",
            "  ─── P&L SUMMARY ────────────────────────────────────────",
            f"  Total Gross P&L : ${m['total_pnl_gross']:>12,.2f}",
            f"  Total Costs     : ${m['total_cost']:>12,.2f}",
            f"  Total Net P&L   : ${m['total_pnl_net']:>12,.2f}",
            f"  Cost drag       : {m['cost_bps_per_yr']:.1f} bps / yr",
            "",
            "  ─── TRADING ACTIVITY ───────────────────────────────────",
            f"  Avg Daily Turnover : {m['avg_daily_turnover']:.2%} of notional",
            f"  Avg Gross Exposure : ${m['avg_gross_exposure']:,.0f}",
            f"  Total Traded       : ${m['total_traded']:,.0f}",
            f"  Win Rate (daily)   : {m['win_rate']:.1%}",
            "=" * 62,
            "",
        ]
        text = "\n".join(lines)
        print(text)
        (RESULTS_DIR / "tearsheet.txt").write_text(text)

    def _save_metrics(self, metrics: dict):
        path = RESULTS_DIR / "metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved → {path}")
