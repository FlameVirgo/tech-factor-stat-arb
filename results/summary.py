"""
summary.py — Print a comprehensive metrics summary for the OU Stat-Arb backtest.

Run from the results/ directory:
    python summary.py
"""

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent


def _load_metrics() -> dict:
    path = RESULTS_DIR / "metrics.json"
    if not path.exists():
        print("ERROR: metrics.json not found. Run the backtest first (python main.py).")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def _grade(sharpe_net: float, mdd: float, win_rate: float) -> str:
    if sharpe_net >= 1.0 and mdd > -0.10:
        return "A  (strong)"
    if sharpe_net >= 0.5 and mdd > -0.20:
        return "B  (acceptable)"
    if sharpe_net >= 0.0:
        return "C  (marginal)"
    return "D  (loss-making net of costs)"


def print_summary(m: dict) -> None:
    w = 62

    def sep(ch="─"):
        print(ch * w)

    def row(label, value, indent=2):
        pad = " " * indent
        print(f"{pad}{label:<38}{value}")

    sep("=")
    print("  OU STAT-ARB (PCA-LW) — COMPREHENSIVE METRICS SUMMARY")
    sep("=")

    # ── Universe & Period ──────────────────────────────────────────────
    print()
    print("  UNIVERSE & PERIOD")
    sep()
    row("Period",       f"{m['date_start']}  →  {m['date_end']}")
    row("Duration",     f"{m['years']:.1f} years  ({m['n_trading_days']:,} trading days)")
    row("Universe",     "50 US tech stocks (39 retained after cleaning)")
    row("Notional",     "$1,000,000")

    # ── Return Metrics ─────────────────────────────────────────────────
    print()
    print("  RETURN METRICS")
    sep()
    gn = m["ann_return_gross"]
    nn = m["ann_return_net"]
    row("Ann. Return (gross)",  f"{gn:+.2%}")
    row("Ann. Return (net)",    f"{nn:+.2%}")
    row("Ann. Volatility",      f"{m['ann_vol_net']:.2%}")
    row("Total Gross P&L",      f"${m['total_pnl_gross']:>14,.2f}")
    row("Total Net P&L",        f"${m['total_pnl_net']:>14,.2f}")

    # ── Risk-Adjusted ──────────────────────────────────────────────────
    print()
    print("  RISK-ADJUSTED METRICS")
    sep()
    row("Sharpe  (gross)",  f"{m['sharpe_gross']:>+.3f}")
    row("Sharpe  (net)",    f"{m['sharpe_net']:>+.3f}")
    row("Sortino (net)",    f"{m['sortino_net']:>+.3f}")
    calmar = m.get("calmar_ratio")
    row("Calmar  (net)",    f"{calmar:>+.3f}" if calmar else "N/A")
    row("Max Drawdown",     f"{m['max_drawdown']:>+.2%}")

    # ── Cost Analysis ──────────────────────────────────────────────────
    print()
    print("  COST ANALYSIS")
    sep()
    row("Total Costs",          f"${m['total_cost']:>14,.2f}")
    row("Cost Drag",            f"{m['cost_bps_per_yr']:.1f} bps / year")
    row("Gross-to-Net Gap",     f"{(gn - nn):+.2%}  (costs ate {abs(gn - nn) / max(abs(gn), 1e-9):.0%} of gross)")
    row("Avg Daily Turnover",   f"{m['avg_daily_turnover']:.2%} of notional")
    row("Avg Gross Exposure",   f"${m['avg_gross_exposure']:>12,.0f}")
    row("Total Dollar Traded",  f"${m['total_traded']:>14,.0f}")

    # ── Trading Activity ───────────────────────────────────────────────
    print()
    print("  TRADING ACTIVITY")
    sep()
    row("Win Rate (daily P&L)", f"{m['win_rate']:.1%}")

    # ── Flags & Verdict ────────────────────────────────────────────────
    print()
    print("  FLAGS")
    sep()
    flags = []
    if m["sharpe_net"] < 0:
        flags.append("[!] Sharpe net < 0  — strategy loses money after costs")
    if m["cost_bps_per_yr"] > 150:
        flags.append(f"[!] Cost drag {m['cost_bps_per_yr']:.0f} bps/yr — consider reducing turnover")
    if m["max_drawdown"] < -0.15:
        flags.append(f"[!] Max drawdown {m['max_drawdown']:.1%} — tail risk is elevated")
    if m["win_rate"] < 0.45:
        flags.append(f"[!] Win rate {m['win_rate']:.1%} — less than half of days profitable")
    if not flags:
        flags.append("[OK] No critical flags raised.")
    for f in flags:
        print(f"  {f}")

    print()
    print("  OVERALL GRADE")
    sep()
    grade = _grade(m["sharpe_net"], m["max_drawdown"], m["win_rate"])
    print(f"  {grade}")
    print()
    print("  Key diagnosis: gross Sharpe {:.3f} shows the signal works,".format(m["sharpe_gross"]))
    print(f"  but transaction costs ({m['cost_bps_per_yr']:.0f} bps/yr) overwhelm the edge.")
    print("  Reduce rebalance frequency or market impact assumption to improve.")
    print()
    sep("=")

    # ── File locations ─────────────────────────────────────────────────
    print()
    print("  OUTPUT FILES  (results/)")
    sep()
    files = [
        ("tearsheet.txt",              "One-page tearsheet"),
        ("metrics.json",               "All metrics as JSON"),
        ("00_equity_curve.png",        "Cumulative P&L curve"),
        ("01_drawdown.png",            "Drawdown over time"),
        ("02_monthly_returns_heatmap.png", "Monthly return heatmap"),
        ("03_rolling_sharpe.png",      "Rolling 252-day Sharpe"),
        ("04_cost_breakdown.png",      "Gross vs net vs costs"),
        ("05_positions_heatmap.png",   "Position heatmap by ticker"),
        ("run.log",                    "Full run log"),
    ]
    for fname, desc in files:
        exists = "✓" if (RESULTS_DIR / fname).exists() else "✗"
        row(f"{exists}  {fname}", desc)
    print()


if __name__ == "__main__":
    m = _load_metrics()
    print_summary(m)
