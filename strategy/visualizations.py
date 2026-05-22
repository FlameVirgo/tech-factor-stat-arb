"""
visualizations.py — 10 research-grade charts for the cross-sectional stat-arb strategy.

CHARTS PRODUCED:
    01_cumulative_pnl.png         — Net PnL vs SPY buy-and-hold benchmark
    02_rolling_sharpe.png         — 63-day rolling Sharpe ratio
    03_drawdown.png               — Underwater equity curve (drawdown chart)
    04_factor_exposure.png        — Portfolio factor exposure over time (stacked)
    05_quintile_spread.png        — Average return by signal quintile
    06_ic_over_time.png           — Rolling 21-day IC with full-period mean
    07_turnover.png               — Daily portfolio turnover % of NAV
    08_pnl_attribution.png        — Factor vs idiosyncratic PnL (stacked)
    09_correlation_heatmap.png    — Return correlation before/after neutralization
    10_trade_pnl_histogram.png    — Distribution of daily PnL

All charts use a clean, publication-quality style (dark grid, muted palette).
Files are saved to results/ and the path is returned for logging.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe in scripts

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from config import CONFIG

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Style defaults applied once at module load
# ─────────────────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    "figure.dpi":       150,
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f9fa",
    "axes.edgecolor":   "#cccccc",
    "axes.labelsize":   11,
    "axes.titlesize":   13,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "lines.linewidth":  1.4,
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
COLORS = {
    "strategy":    "#2196F3",   # blue  — strategy line
    "benchmark":   "#FF9800",   # orange — SPY
    "positive":    "#4CAF50",   # green
    "negative":    "#F44336",   # red
    "drawdown":    "#F44336",   # red fill
    "neutral":     "#9E9E9E",   # grey
    "factors": [
        "#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"
    ],
}


class Visualizer:
    """
    Generate all 10 research charts and save to results/.

    Parameters
    ----------
    daily_pnl        : pd.Series     Net daily P&L in dollars.
    gross_pnl        : pd.Series     Pre-cost daily P&L in dollars.
    cost_series      : pd.Series     Transaction + borrow costs per day.
    book_stats       : pd.DataFrame  Output of Backtester._compute_book_stats().
    factor_exp_ts    : pd.DataFrame  Portfolio factor exposures over time.
    signals          : pd.DataFrame  Alpha signal (dates × tickers).
    residuals        : pd.DataFrame  Raw idiosyncratic returns (dates × tickers).
    trades           : pd.DataFrame  Dollar trades each day (dates × tickers).
    metrics          : dict          Full output of PerformanceAnalytics.compute_all().
    spy_returns      : pd.Series     SPY daily log returns (for benchmark).
    config           : dict          CONFIG from config.py.
    """

    def __init__(
        self,
        daily_pnl:     pd.Series,
        gross_pnl:     pd.Series,
        cost_series:   pd.Series,
        book_stats:    pd.DataFrame,
        factor_exp_ts: pd.DataFrame,
        signals:       pd.DataFrame,
        residuals:     pd.DataFrame,
        trades:        pd.DataFrame,
        metrics:       dict,
        spy_returns:   pd.Series,
        config:        dict = CONFIG,
    ):
        self.daily_pnl     = daily_pnl
        self.gross_pnl     = gross_pnl
        self.cost_series   = cost_series
        self.book_stats    = book_stats
        self.factor_exp_ts = factor_exp_ts
        self.signals       = signals
        self.residuals     = residuals
        self.trades        = trades
        self.metrics       = metrics
        self.spy_returns   = spy_returns
        self.cfg           = config

        os.makedirs(RESULTS_DIR, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def plot_all(self) -> list:
        """
        Generate all 10 charts and return list of saved file paths.
        """
        logger.info("=== Visualizations ===")
        paths = []
        plots = [
            self._plot_cumulative_pnl,
            self._plot_rolling_sharpe,
            self._plot_drawdown,
            self._plot_factor_exposure,
            self._plot_quintile_spread,
            self._plot_ic_over_time,
            self._plot_turnover,
            self._plot_pnl_attribution,
            self._plot_correlation_heatmap,
            self._plot_trade_pnl_histogram,
        ]
        for fn in plots:
            try:
                path = fn()
                paths.append(path)
                logger.info(f"  Saved: {os.path.basename(path)}")
            except Exception as e:
                logger.warning(f"  Chart {fn.__name__} failed: {e}")
        return paths

    # ─────────────────────────────────────────────────────────────────────────
    # CHART 1 — Cumulative PnL vs SPY
    # ─────────────────────────────────────────────────────────────────────────

    def _plot_cumulative_pnl(self) -> str:
        """
        Cumulative net P&L in dollars vs a SPY buy-and-hold benchmark.

        The SPY benchmark is scaled to the same starting notional ($100k) so
        both curves start at $0 and show dollar profit/loss over time.
        This lets us directly compare strategy edge vs passive index exposure.

        Walk-forward period boundaries are shown as vertical dashed lines.
        """
        fig, ax = plt.subplots(figsize=(12, 5))

        # Strategy cumulative PnL (in dollars)
        cum_pnl = self.daily_pnl.cumsum()
        ax.plot(cum_pnl.index, cum_pnl.values, color=COLORS["strategy"],
                label="Strategy (net)", linewidth=1.5, zorder=3)

        # SPY benchmark: scale a $100k investment
        spy = self.spy_returns.reindex(self.daily_pnl.index).fillna(0)
        nav = self.cfg["portfolio_notional"]
        spy_cum = nav * (np.exp(spy.cumsum()) - 1)   # dollar profit on $100k
        ax.plot(spy_cum.index, spy_cum.values, color=COLORS["benchmark"],
                label="SPY buy-and-hold", linewidth=1.2, linestyle="--", zorder=2)

        # Walk-forward period shading
        self._shade_periods(ax, self.daily_pnl.index)

        ax.axhline(0, color="black", linewidth=0.7, linestyle="-", zorder=1)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.set_title("Cumulative Net P&L vs SPY Benchmark", fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative P&L ($)")
        ax.legend(loc="upper left")
        fig.tight_layout()

        path = os.path.join(RESULTS_DIR, "01_cumulative_pnl.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─────────────────────────────────────────────────────────────────────────
    # CHART 2 — Rolling 63-day Sharpe Ratio
    # ─────────────────────────────────────────────────────────────────────────

    def _plot_rolling_sharpe(self) -> str:
        """
        Rolling 63-day (≈ 3-month) annualised Sharpe ratio of net daily P&L.

        A rolling Sharpe shows whether the strategy's risk-adjusted performance
        is stable over time or deteriorates in certain regimes.  Sustained periods
        below 0 indicate the signal has stopped working.
        """
        fig, ax = plt.subplots(figsize=(12, 4))

        window = 63
        tdy    = self.cfg["trading_days_per_year"]

        roll_mean = self.daily_pnl.rolling(window, min_periods=window // 2).mean()
        roll_std  = self.daily_pnl.rolling(window, min_periods=window // 2).std()
        roll_sharpe = (roll_mean / roll_std.replace(0, np.nan)) * np.sqrt(tdy)

        # Colour positive/negative regions
        ax.fill_between(roll_sharpe.index, roll_sharpe.values, 0,
                        where=(roll_sharpe >= 0), interpolate=True,
                        color=COLORS["positive"], alpha=0.3, label="Positive")
        ax.fill_between(roll_sharpe.index, roll_sharpe.values, 0,
                        where=(roll_sharpe < 0), interpolate=True,
                        color=COLORS["negative"], alpha=0.3, label="Negative")
        ax.plot(roll_sharpe.index, roll_sharpe.values,
                color=COLORS["strategy"], linewidth=1.2)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(1, color=COLORS["positive"], linewidth=0.6, linestyle=":")
        ax.axhline(-1, color=COLORS["negative"], linewidth=0.6, linestyle=":")

        self._shade_periods(ax, self.daily_pnl.index)

        ax.set_title(f"Rolling {window}-Day Annualised Sharpe Ratio", fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sharpe Ratio")
        ax.legend(loc="upper left")
        fig.tight_layout()

        path = os.path.join(RESULTS_DIR, "02_rolling_sharpe.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─────────────────────────────────────────────────────────────────────────
    # CHART 3 — Drawdown (Underwater Equity Curve)
    # ─────────────────────────────────────────────────────────────────────────

    def _plot_drawdown(self) -> str:
        """
        Underwater equity curve: the running drawdown from the highest cumulative PnL.

        A drawdown of -$5,000 means the strategy is $5,000 below its historical peak.
        The maximum drawdown (MDD) is the deepest trough — a key risk metric.
        Shallow, short-lived drawdowns indicate robust alpha; prolonged deep ones
        suggest regime change or signal decay.
        """
        fig, ax = plt.subplots(figsize=(12, 4))

        cum_pnl   = self.daily_pnl.cumsum()
        peak      = cum_pnl.expanding().max()
        drawdown  = cum_pnl - peak   # always ≤ 0

        ax.fill_between(drawdown.index, drawdown.values, 0,
                        color=COLORS["drawdown"], alpha=0.5, label="Drawdown")
        ax.plot(drawdown.index, drawdown.values,
                color=COLORS["negative"], linewidth=0.8)

        # Mark the maximum drawdown point
        mdd_idx = drawdown.idxmin()
        mdd_val = drawdown.min()
        ax.annotate(
            f"MDD: ${mdd_val:,.0f}",
            xy=(mdd_idx, mdd_val),
            xytext=(mdd_idx, mdd_val * 0.6),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
            fontsize=9, color="black",
        )

        self._shade_periods(ax, self.daily_pnl.index)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.set_title("Drawdown (Underwater Equity Curve)", fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Drawdown from Peak ($)")
        ax.legend(loc="lower left")
        fig.tight_layout()

        path = os.path.join(RESULTS_DIR, "03_drawdown.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─────────────────────────────────────────────────────────────────────────
    # CHART 4 — Factor Exposure Over Time
    # ─────────────────────────────────────────────────────────────────────────

    def _plot_factor_exposure(self) -> str:
        """
        Portfolio net exposure to each of the 5 risk factors over time.

        A well-constructed factor-neutral portfolio should have exposures
        near zero for all factors.  Persistent tilts to beta or momentum
        mean the strategy is inadvertently capturing a known risk premium
        rather than pure idiosyncratic alpha.

        The dashed lines at ±0.2 show the warning threshold from CONFIG.
        """
        fig, ax = plt.subplots(figsize=(12, 5))

        factors = self.factor_exp_ts.dropna(how="all")
        if factors.empty:
            ax.text(0.5, 0.5, "No factor exposure data", transform=ax.transAxes,
                    ha="center", va="center")
        else:
            for i, col in enumerate(factors.columns):
                color = COLORS["factors"][i % len(COLORS["factors"])]
                smoothed = factors[col].rolling(21, min_periods=5).mean()
                ax.plot(smoothed.index, smoothed.values, label=col.capitalize(),
                        color=color, linewidth=1.2)

        threshold = self.cfg.get("max_factor_exposure", 0.2)
        ax.axhline( threshold, color="grey", linewidth=0.7, linestyle=":", alpha=0.8)
        ax.axhline(-threshold, color="grey", linewidth=0.7, linestyle=":", alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.6)

        self._shade_periods(ax, self.factor_exp_ts.index)

        ax.set_title("Portfolio Factor Exposure Over Time (21-Day Smoothed)", fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Net Factor Exposure (fraction of NAV)")
        ax.legend(loc="upper right", ncol=3)
        fig.tight_layout()

        path = os.path.join(RESULTS_DIR, "04_factor_exposure.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─────────────────────────────────────────────────────────────────────────
    # CHART 5 — Quintile Return Spread
    # ─────────────────────────────────────────────────────────────────────────

    def _plot_quintile_spread(self) -> str:
        """
        Average NEXT-DAY return for each signal quintile (Q1=lowest, Q5=highest signal).

        A monotonically increasing or decreasing pattern confirms the signal ranks
        stocks correctly relative to forward returns.  The spread between Q5 and Q1
        (the "quintile spread") is the raw edge being exploited.

        Note: this chart uses future returns and is for ANALYSIS ONLY, not signal
        construction.  It shows whether the signal has predictive power in-sample.
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        # 1-day forward return for each stock-date
        fwd_ret = self.residuals.shift(-1)   # next day's idiosyncratic return

        # Align signal and forward return, stack to (date, ticker) pairs
        sig_stack = self.signals.stack().rename("signal")
        fwd_stack = fwd_ret.stack().rename("fwd_ret")
        combined  = pd.concat([sig_stack, fwd_stack], axis=1).dropna()

        # Assign quintiles by cross-sectional signal rank each day
        combined["quintile"] = combined.groupby(level=0)["signal"].transform(
            lambda x: pd.qcut(x, q=5, labels=[1, 2, 3, 4, 5], duplicates="drop")
        )
        combined = combined.dropna(subset=["quintile"])
        combined["quintile"] = combined["quintile"].astype(int)

        # Mean forward return per quintile (in percent, annualised)
        tdy = self.cfg["trading_days_per_year"]
        quintile_means = combined.groupby("quintile")["fwd_ret"].mean() * tdy * 100

        colors = [
            COLORS["negative"] if v < 0 else COLORS["positive"]
            for v in quintile_means.values
        ]
        bars = ax.bar(quintile_means.index, quintile_means.values,
                      color=colors, edgecolor="white", linewidth=0.5)

        # Annotate bars
        for bar, val in zip(bars, quintile_means.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (0.01 if val >= 0 else -0.05),
                f"{val:.2f}%",
                ha="center", va="bottom" if val >= 0 else "top",
                fontsize=9
            )

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(["Q1\n(Lowest)", "Q2", "Q3", "Q4", "Q5\n(Highest)"])
        ax.set_title("Quintile Return Spread (Annualised % by Signal Quintile)", fontweight="bold")
        ax.set_xlabel("Signal Quintile")
        ax.set_ylabel("Mean Forward Return (annualised %)")
        fig.tight_layout()

        path = os.path.join(RESULTS_DIR, "05_quintile_spread.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─────────────────────────────────────────────────────────────────────────
    # CHART 6 — IC Over Time
    # ─────────────────────────────────────────────────────────────────────────

    def _plot_ic_over_time(self) -> str:
        """
        Rolling 21-day average Information Coefficient (IC) over time.

        IC = Spearman rank correlation between signal_t and next-day return.
        A positive IC means the signal ranks stocks correctly relative to future
        returns.  The rolling 21-day average smooths day-to-day noise and shows
        whether IC is stable or decaying (signal degradation).

        The horizontal dashed line is the full-period mean IC — a benchmark.
        """
        fig, ax = plt.subplots(figsize=(12, 4))

        # Compute daily IC
        fwd_ret = self.residuals.shift(-1)
        from scipy.stats import spearmanr

        ic_vals, ic_dates = [], []
        for date in self.signals.index:
            s = self.signals.loc[date].dropna()
            if date not in fwd_ret.index:
                continue
            f = fwd_ret.loc[date].reindex(s.index).dropna()
            common = s.index.intersection(f.index)
            if len(common) < 10:
                continue
            corr, _ = spearmanr(s.loc[common].values, f.loc[common].values)
            ic_vals.append(corr)
            ic_dates.append(date)

        if not ic_dates:
            ax.text(0.5, 0.5, "Insufficient data for IC", transform=ax.transAxes,
                    ha="center", va="center")
        else:
            daily_ic = pd.Series(ic_vals, index=pd.DatetimeIndex(ic_dates))
            rolling_ic = daily_ic.rolling(21, min_periods=10).mean()

            ax.fill_between(rolling_ic.index, rolling_ic.values, 0,
                            where=(rolling_ic >= 0), interpolate=True,
                            color=COLORS["positive"], alpha=0.35)
            ax.fill_between(rolling_ic.index, rolling_ic.values, 0,
                            where=(rolling_ic < 0), interpolate=True,
                            color=COLORS["negative"], alpha=0.35)
            ax.plot(rolling_ic.index, rolling_ic.values,
                    color=COLORS["strategy"], linewidth=1.2, label="21-day rolling IC")

            mean_ic = daily_ic.mean()
            ax.axhline(mean_ic, color="navy", linewidth=0.8, linestyle="--",
                       label=f"Mean IC = {mean_ic:.4f}")
            ax.axhline(0, color="black", linewidth=0.7)

            self._shade_periods(ax, daily_ic.index)

        ax.set_title("Rolling 21-Day Information Coefficient (IC)", fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("IC (Spearman rank correlation)")
        ax.legend(loc="upper right")
        fig.tight_layout()

        path = os.path.join(RESULTS_DIR, "06_ic_over_time.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─────────────────────────────────────────────────────────────────────────
    # CHART 7 — Turnover Over Time
    # ─────────────────────────────────────────────────────────────────────────

    def _plot_turnover(self) -> str:
        """
        Daily portfolio turnover as a percentage of NAV.

        Turnover = sum(|trades_i|) / NAV × 100%.
        High turnover is a key cost driver — each round-trip has ~20bps of
        friction ($0.005/share + 10bps market impact × 2 sides).
        A weekly-rebalance strategy typically targets 5-10% daily turnover.
        The turnover filter in PortfolioConstructor directly reduces this.
        """
        fig, ax = plt.subplots(figsize=(12, 4))

        turnover_pct = self.trades.abs().sum(axis=1) / self.cfg["portfolio_notional"] * 100
        rolling_turn = turnover_pct.rolling(21, min_periods=5).mean()

        ax.bar(turnover_pct.index, turnover_pct.values,
               color=COLORS["neutral"], alpha=0.4, width=1, label="Daily turnover")
        ax.plot(rolling_turn.index, rolling_turn.values,
                color=COLORS["strategy"], linewidth=1.4, label="21-day average")

        ax.set_title("Daily Portfolio Turnover (% of NAV)", fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Turnover (%)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        ax.legend(loc="upper right")
        self._shade_periods(ax, turnover_pct.index)
        fig.tight_layout()

        path = os.path.join(RESULTS_DIR, "07_turnover.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─────────────────────────────────────────────────────────────────────────
    # CHART 8 — PnL Attribution: Factor vs Idiosyncratic
    # ─────────────────────────────────────────────────────────────────────────

    def _plot_pnl_attribution(self) -> str:
        """
        Decompose daily gross P&L into factor PnL and idiosyncratic PnL.

        factor_pnl = sum_k(portfolio_exp_k × factor_return_k × NAV)
        idio_pnl   = gross_pnl − factor_pnl

        For a factor-neutral strategy, the vast majority of P&L should come from
        the idiosyncratic (alpha) component.  If factor PnL dominates, the strategy
        is really just a disguised beta/momentum bet and is not true stat-arb.

        Uses a 21-day rolling cumulative sum for readability (daily bars are too noisy).
        """
        fig, ax = plt.subplots(figsize=(12, 5))

        risk = self.metrics.get("risk", {})
        factor_pnl_series = risk.get("factor_pnl_series")
        idio_pnl_series   = risk.get("idio_pnl_series")

        if factor_pnl_series is None or idio_pnl_series is None:
            # Fall back: show gross vs net
            cum_gross = self.gross_pnl.cumsum()
            cum_net   = self.daily_pnl.cumsum()
            ax.plot(cum_gross.index, cum_gross.values,
                    color=COLORS["benchmark"], label="Gross PnL", linewidth=1.2)
            ax.plot(cum_net.index, cum_net.values,
                    color=COLORS["strategy"], label="Net PnL", linewidth=1.2)
            ax.set_title("Gross vs Net Cumulative P&L", fontweight="bold")
        else:
            # Stacked area: factor vs idiosyncratic cumulative PnL
            # factor_pnl_series may be a DataFrame (dates × factors); sum across factors
            if hasattr(factor_pnl_series, "ndim") and factor_pnl_series.ndim == 2:
                factor_total = factor_pnl_series.sum(axis=1)
            else:
                factor_total = pd.Series(factor_pnl_series)
            factor_cum = factor_total.sort_index().cumsum()
            idio_cum   = pd.Series(idio_pnl_series).sort_index().cumsum()
            common_idx = factor_cum.index.intersection(idio_cum.index)
            factor_cum = factor_cum.reindex(common_idx)
            idio_cum   = idio_cum.reindex(common_idx)

            ax.fill_between(common_idx, 0, idio_cum.values,
                            alpha=0.55, color=COLORS["strategy"], label="Idiosyncratic PnL")
            ax.fill_between(common_idx, idio_cum.values,
                            idio_cum.values + factor_cum.values,
                            alpha=0.55, color=COLORS["benchmark"], label="Factor PnL")
            ax.plot(common_idx, (idio_cum + factor_cum).values,
                    color="black", linewidth=0.8, label="Total gross PnL")
            ax.set_title("Cumulative P&L Attribution: Idiosyncratic vs Factor", fontweight="bold")

        ax.axhline(0, color="black", linewidth=0.7)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative P&L ($)")
        ax.legend(loc="upper left")
        self._shade_periods(ax, self.daily_pnl.index)
        fig.tight_layout()

        path = os.path.join(RESULTS_DIR, "08_pnl_attribution.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─────────────────────────────────────────────────────────────────────────
    # CHART 9 — Return Correlation Heatmap (before vs after factor neutralization)
    # ─────────────────────────────────────────────────────────────────────────

    def _plot_correlation_heatmap(self) -> str:
        """
        Cross-stock return correlation heatmap: raw returns vs idiosyncratic residuals.

        Raw tech stock returns are highly correlated — they move together when the
        market rallies or sells off.  After Fama-MacBeth factor neutralization, the
        residual idiosyncratic returns should be much less correlated, confirming
        that the common factor structure has been successfully removed.

        A more diagonal (lower off-diagonal) heatmap in the right panel is the goal.
        """
        # Subsample tickers to keep the plot legible (max 20)
        tickers = list(self.signals.columns[:20])

        raw_corr  = self.signals.reindex(columns=tickers).corr()   # proxy: use signals
        idio_corr = self.residuals.reindex(columns=tickers).corr()

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for ax, corr, title in zip(
            axes,
            [raw_corr, idio_corr],
            ["Signal Correlation\n(before neutralization)", "Idiosyncratic Return Correlation\n(after neutralization)"],
        ):
            mask = np.zeros_like(corr, dtype=bool)
            np.fill_diagonal(mask, True)
            sns.heatmap(
                corr, ax=ax, mask=mask,
                cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.3, linecolor="white",
                cbar_kws={"shrink": 0.7},
                xticklabels=tickers, yticklabels=tickers,
                annot=False,
            )
            ax.set_title(title, fontweight="bold", fontsize=11)
            ax.tick_params(axis="x", labelsize=7, rotation=45)
            ax.tick_params(axis="y", labelsize=7, rotation=0)

        fig.suptitle("Cross-Stock Correlation: Before vs After Factor Neutralization",
                     fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()

        path = os.path.join(RESULTS_DIR, "09_correlation_heatmap.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─────────────────────────────────────────────────────────────────────────
    # CHART 10 — Daily PnL Distribution (histogram)
    # ─────────────────────────────────────────────────────────────────────────

    def _plot_trade_pnl_histogram(self) -> str:
        """
        Distribution of daily net P&L in dollars.

        A strategy with positive alpha should have:
          - Mean > 0 (positive expected daily return)
          - Moderate kurtosis (fat tails indicate rare large losses)
          - More days above 0 than below (win rate)

        The vertical lines show the mean (target: positive) and ±1 std dev.
        """
        fig, ax = plt.subplots(figsize=(9, 5))

        pnl = self.daily_pnl.dropna().values
        mean_pnl = pnl.mean()
        std_pnl  = pnl.std()

        # Histogram with KDE overlay
        n_bins = min(80, len(pnl) // 5)
        ax.hist(pnl, bins=n_bins, color=COLORS["strategy"], alpha=0.6, density=True,
                edgecolor="white", linewidth=0.3, label="Daily PnL distribution")

        # KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(pnl, bw_method="scott")
        x   = np.linspace(pnl.min(), pnl.max(), 300)
        ax.plot(x, kde(x), color="navy", linewidth=1.5, label="KDE")

        # Vertical reference lines
        ax.axvline(mean_pnl, color=COLORS["positive"], linewidth=1.4, linestyle="--",
                   label=f"Mean = ${mean_pnl:,.1f}")
        ax.axvline(mean_pnl + std_pnl, color=COLORS["neutral"], linewidth=0.8,
                   linestyle=":", label=f"±1 std (${std_pnl:,.0f})")
        ax.axvline(mean_pnl - std_pnl, color=COLORS["neutral"], linewidth=0.8, linestyle=":")
        ax.axvline(0, color="black", linewidth=0.8)

        # Win rate annotation
        win_rate = (pnl > 0).mean() * 100
        ax.text(0.97, 0.95, f"Win rate: {win_rate:.1f}%", transform=ax.transAxes,
                ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.set_title("Daily Net P&L Distribution", fontweight="bold")
        ax.set_xlabel("Daily Net P&L ($)")
        ax.set_ylabel("Density")
        ax.legend(loc="upper left")
        fig.tight_layout()

        path = os.path.join(RESULTS_DIR, "10_trade_pnl_histogram.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    # ─────────────────────────────────────────────────────────────────────────
    # HELPER
    # ─────────────────────────────────────────────────────────────────────────

    def _shade_periods(self, ax: plt.Axes, idx: pd.DatetimeIndex) -> None:
        """
        Add light vertical shading and labels for each walk-forward period.

        Train (2015-2020)   = light blue
        Validation (2021-2022) = light orange
        Test (2023+)        = light green

        Parameters
        ----------
        ax  : matplotlib Axes object.
        idx : DatetimeIndex of the data being plotted (used to clip period bounds).
        """
        cfg = self.cfg
        periods = [
            (cfg["start_date"],  cfg["train_end"],  "#2196F3", "Train"),
            (cfg["val_start"],   cfg["val_end"],    "#FF9800", "Val"),
            (cfg["test_start"],  str(idx[-1].date()), "#4CAF50", "Test"),
        ]
        ylim = ax.get_ylim()
        for start, end, color, label in periods:
            try:
                ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                           alpha=0.06, color=color, zorder=0)
                # Add period label at the top of the span
                mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
                ax.text(mid, ylim[1], label, ha="center", va="top",
                        fontsize=7, color=color, alpha=0.7)
            except Exception:
                pass
        ax.set_ylim(ylim)
