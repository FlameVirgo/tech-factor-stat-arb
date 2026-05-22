"""
results/summary.py — Standalone tearsheet for the cross-sectional tech stat-arb strategy.

Run this after main.py has produced results/metrics.json:
    python results/summary.py

OUTPUT:
    1. Colour-coded console tearsheet (matching the pairs trading format).
    2. results/00_summary_table.png  — publication-quality metrics table.

The script is intentionally standalone — no imports from strategy/ — so it
can be run independently to inspect results without re-running the full pipeline.
"""

import datetime
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")

# ANSI colour codes for terminal output
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def _load_metrics() -> dict:
    if not os.path.exists(METRICS_PATH):
        print(f"{RED}ERROR: metrics.json not found at {METRICS_PATH}{RESET}")
        print("Run  python main.py  first to generate results.")
        sys.exit(1)
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


def _fmt(value, fmt=".2f", prefix="", suffix="", missing="N/A"):
    """Safely format a number, returning 'N/A' if None or NaN."""
    if value is None:
        return missing
    try:
        v = float(value)
        if v != v:   # NaN check
            return missing
        return f"{prefix}{v:{fmt}}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _colour(value, good_positive=True, threshold=0.0):
    """Return green if value is on the 'good' side, red otherwise."""
    try:
        v = float(value)
        if v != v:
            return YELLOW
        if good_positive:
            return GREEN if v > threshold else RED
        else:
            return GREEN if v < threshold else RED
    except (TypeError, ValueError):
        return RESET


def _print_line(label, value_str, colour=RESET, width=38):
    print(f"  {label:<{width}} {colour}{value_str}{RESET}")


def _separator(char="─", width=60):
    print(CYAN + char * width + RESET)


# ─────────────────────────────────────────────────────────────────────────────
# Console tearsheet
# ─────────────────────────────────────────────────────────────────────────────

def print_tearsheet(m: dict) -> None:
    full   = m.get("full", {})
    train  = m.get("train", {})
    val    = m.get("val", {})
    test   = m.get("test", {})
    signal = m.get("signal", {})
    risk   = m.get("risk", {})
    costs  = m.get("costs", {})
    regime = m.get("regimes", {})
    warnings = m.get("warnings", [])
    meta   = m.get("meta", {})

    # ── Header ───────────────────────────────────────────────────────────────
    print()
    print(BOLD + CYAN + "=" * 60 + RESET)
    print(BOLD + "   CROSS-SECTIONAL TECH STAT-ARB  —  BACKTEST TEARSHEET" + RESET)
    print(BOLD + CYAN + "=" * 60 + RESET)
    window_start = meta.get("start_date", "2015-01-01")
    window_end   = meta.get("end_date",   str(datetime.date.today()))
    print(f"  {BOLD}Universe:{RESET}  {meta.get('n_stocks', 50)} tech stocks")
    print(f"  {BOLD}Window:{RESET}    {window_start}  →  {window_end}")
    print(f"  {BOLD}Capital:{RESET}   $100,000 NAV   |   $50k long / $50k short")
    _separator()

    # ── Full Period Performance ───────────────────────────────────────────────
    print(BOLD + "\n  FULL-PERIOD PERFORMANCE" + RESET)
    total_pnl = full.get("total_pnl")
    _print_line("Total P&L",
                _fmt(total_pnl, ",.2f", prefix="$"),
                _colour(total_pnl))
    ann_ret = full.get("ann_return")
    _print_line("Annualised Return",
                _fmt(ann_ret, "+.2%"),
                _colour(ann_ret))
    sharpe = full.get("sharpe")
    _print_line("Annualised Sharpe",
                _fmt(sharpe, ".3f"),
                _colour(sharpe, threshold=0.5))
    sortino = full.get("sortino")
    _print_line("Sortino Ratio",
                _fmt(sortino, ".3f"),
                _colour(sortino, threshold=0.7))
    # max_drawdown is stored as a fraction (e.g. -0.05 = -5%); convert to dollars
    mdd_frac = full.get("max_drawdown")
    mdd_usd  = mdd_frac * 100_000 if mdd_frac is not None else None
    _print_line("Max Drawdown",
                _fmt(mdd_usd, ",.2f", prefix="$"),
                _colour(mdd_usd, good_positive=False))
    mdd_days = full.get("mdd_duration")
    _print_line("  → Duration",
                _fmt(mdd_days, ".0f", suffix=" days"))
    calmar = full.get("calmar")
    _print_line("Calmar Ratio",
                _fmt(calmar, ".3f"),
                _colour(calmar, threshold=0.5))
    win_d = full.get("win_rate_daily")
    _print_line("Daily Win Rate",
                _fmt(win_d, ".1%"),
                _colour(win_d, threshold=0.50))
    win_w = full.get("win_rate_weekly")
    _print_line("Weekly Win Rate",
                _fmt(win_w, ".1%"),
                _colour(win_w, threshold=0.50))

    # ── Walk-Forward Breakdown ────────────────────────────────────────────────
    _separator()
    print(BOLD + "\n  WALK-FORWARD PERIOD BREAKDOWN" + RESET)
    header = f"  {'Period':<18} {'Sharpe':>8} {'Ann Ret':>10} {'MDD':>14}"
    print(YELLOW + header + RESET)
    print(YELLOW + "  " + "-" * 52 + RESET)

    for label, period_data in [
        ("Train (2015-20)",  train),
        ("Val   (2021-22)",  val),
        ("Test  (2023+)",    test),
    ]:
        s = _fmt(period_data.get("sharpe"), ".3f")
        r = _fmt(period_data.get("ann_return"), "+.2%")
        mdd_f = period_data.get("max_drawdown")
        mdd_u = mdd_f * 100_000 if mdd_f is not None else None
        d = _fmt(mdd_u, ",.0f", prefix="$")
        s_col = _colour(period_data.get("sharpe"), threshold=0.5)
        r_col = _colour(period_data.get("ann_return"))
        d_col = _colour(mdd_u, good_positive=False)
        print(f"  {label:<18} {s_col}{s:>8}{RESET} {r_col}{r:>10}{RESET} {d_col}{d:>14}{RESET}")

    # ── Signal Quality ────────────────────────────────────────────────────────
    _separator()
    print(BOLD + "\n  SIGNAL QUALITY" + RESET)
    ic_mean = signal.get("ic_mean")
    _print_line("IC (daily, Spearman)",
                _fmt(ic_mean, ".4f"),
                _colour(ic_mean, threshold=0.01))
    ic_std = signal.get("ic_std")
    _print_line("IC Std Dev",
                _fmt(ic_std, ".4f"))
    icir = signal.get("icir")
    _print_line("ICIR (IC / IC-std)",
                _fmt(icir, ".3f"),
                _colour(icir, threshold=0.5))
    ic_t = signal.get("ic_tstat")
    _print_line("IC t-stat",
                _fmt(ic_t, ".2f"),
                _colour(ic_t, threshold=1.96))

    # IC decay table
    decay = signal.get("ic_decay", {})
    if decay:
        print(f"\n  {'IC Decay by Horizon':}")
        print(f"  {'  1d':>6} {'  5d':>6} {'  10d':>7} {'  21d':>7}")
        row = ""
        for h in [1, 5, 10, 21]:
            v = decay.get(h) or decay.get(str(h))
            row += f"  {_fmt(v, '.4f'):>6}"
        print(f"  {row}")

    # Long/short attribution
    long_ret  = signal.get("long_ann_ret")
    short_ret = signal.get("short_ann_ret")
    _print_line("Long leg (ann. return)",
                _fmt(long_ret, "+.2%"),
                _colour(long_ret))
    _print_line("Short leg (ann. return)",
                _fmt(short_ret, "+.2%"),
                _colour(short_ret, good_positive=False))

    # ── Risk ──────────────────────────────────────────────────────────────────
    _separator()
    print(BOLD + "\n  RISK" + RESET)
    spy_beta = risk.get("spy_beta")
    _print_line("SPY Beta",
                _fmt(spy_beta, ".3f"),
                _colour(spy_beta, good_positive=False, threshold=0.1))
    spy_corr = risk.get("spy_correlation")
    _print_line("SPY Correlation",
                _fmt(spy_corr, ".3f"),
                _colour(spy_corr, good_positive=False, threshold=0.2))
    idio_frac = risk.get("idiosyncratic_fraction")
    _print_line("Idiosyncratic PnL fraction",
                _fmt(idio_frac, ".1%"),
                _colour(idio_frac, threshold=0.7))

    # Factor PnL attribution
    factor_attr = risk.get("factor_pnl_breakdown", {})
    if factor_attr:
        print(f"\n  {'Factor PnL Attribution':}")
        for fname, pnl_val in factor_attr.items():
            _print_line(f"  → {fname}", _fmt(pnl_val, ",.0f", prefix="$"))

    # ── Costs ────────────────────────────────────────────────────────────────
    _separator()
    print(BOLD + "\n  COSTS" + RESET)
    gross_sharpe = costs.get("gross_sharpe")
    _print_line("Gross Sharpe",
                _fmt(gross_sharpe, ".3f"),
                _colour(gross_sharpe, threshold=0.5))
    net_sharpe = full.get("sharpe")
    _print_line("Net Sharpe",
                _fmt(net_sharpe, ".3f"),
                _colour(net_sharpe, threshold=0.5))
    cost_drag = costs.get("annual_cost_bps")
    _print_line("Annual cost drag",
                _fmt(cost_drag, ".1f", suffix=" bps"),
                _colour(cost_drag, good_positive=False, threshold=30))
    total_costs_usd = costs.get("total_costs_usd")
    _print_line("Total costs paid",
                _fmt(total_costs_usd, ",.0f", prefix="$"),
                _colour(total_costs_usd, good_positive=False))
    avg_turn = costs.get("avg_daily_turnover")
    _print_line("Avg daily turnover",
                _fmt(avg_turn, ".1%"),
                _colour(avg_turn, good_positive=False, threshold=0.15))

    # ── Regime Analysis ──────────────────────────────────────────────────────
    _separator()
    print(BOLD + "\n  REGIME ANALYSIS" + RESET)
    if regime:
        for rdata in regime:
            if not isinstance(rdata, dict):
                continue
            rname = rdata.get("name", "Unknown")
            s     = _fmt(rdata.get("sharpe"), ".3f")
            ic    = _fmt(rdata.get("ic_mean"), ".4f")
            s_col = _colour(rdata.get("sharpe"), threshold=0.5)
            print(f"  {rname:<28} Sharpe {s_col}{s:>7}{RESET}  |  IC {ic}")

    # ── Warnings ─────────────────────────────────────────────────────────────
    if warnings:
        _separator()
        print(BOLD + "\n  WARNINGS" + RESET)
        for w in warnings:
            print(f"  {YELLOW}⚠  {w}{RESET}")

    # ── Footer ───────────────────────────────────────────────────────────────
    _separator("═")
    generated = meta.get("generated_at", str(datetime.datetime.now())[:19])
    print(f"  Generated: {generated}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib summary table
# ─────────────────────────────────────────────────────────────────────────────

def save_summary_table(m: dict) -> str:
    """
    Render a colour-coded metrics table as a PNG image.

    Saves to results/00_summary_table.png and returns the path.
    The table rows are coloured green/red based on whether each metric
    is 'good' or 'bad' — matching the console output above.
    """
    full   = m.get("full", {})
    train  = m.get("train", {})
    val    = m.get("val", {})
    test   = m.get("test", {})
    signal = m.get("signal", {})
    risk   = m.get("risk", {})
    costs  = m.get("costs", {})

    def f(v, fmt=".3f", pct=False, dollar=False):
        if v is None:
            return "N/A"
        try:
            vv = float(v)
            if vv != vv:
                return "N/A"
            if pct:
                return f"{vv:+.2%}"
            if dollar:
                return f"${vv:,.0f}"
            return f"{vv:{fmt}}"
        except Exception:
            return str(v)

    rows = [
        # (Label, Value, good_if_positive, threshold)
        ("Total P&L",                   f(full.get("total_pnl"), dollar=True),         True,  0),
        ("Annualised Return",            f(full.get("ann_return"), pct=True),           True,  0),
        ("Annualised Sharpe",            f(full.get("sharpe")),                         True,  0.5),
        ("Sortino Ratio",                f(full.get("sortino")),                        True,  0.7),
        ("Max Drawdown",                 f((full.get("max_drawdown") or 0) * 100_000, dollar=True), False, 0),
        ("Calmar Ratio",                 f(full.get("calmar")),                         True,  0.5),
        ("Daily Win Rate",               f(full.get("win_rate_daily"), pct=True),       True,  0.5),
        ("",                             "",                                            None,  None),
        ("Train Sharpe (2015-20)",       f(train.get("sharpe")),                        True,  0.5),
        ("Val Sharpe   (2021-22)",       f(val.get("sharpe")),                          True,  0.5),
        ("Test Sharpe  (2023+)",         f(test.get("sharpe")),                         True,  0.5),
        ("",                             "",                                            None,  None),
        ("IC (mean, Spearman)",          f(signal.get("ic_mean"), ".4f"),               True,  0.01),
        ("ICIR",                         f(signal.get("icir")),                         True,  0.5),
        ("IC t-stat",                    f(signal.get("ic_tstat")),                     True,  1.96),
        ("",                             "",                                            None,  None),
        ("SPY Beta",                     f(risk.get("spy_beta")),                       False, 0.1),
        ("SPY Correlation",              f(risk.get("spy_correlation")),                False, 0.2),
        ("Idiosyncratic PnL fraction",   f(risk.get("idiosyncratic_fraction"), pct=True), True, 0.7),
        ("",                             "",                                            None,  None),
        ("Gross Sharpe",                 f(costs.get("gross_sharpe")),                  True,  0.5),
        ("Annual Cost Drag",             f(costs.get("annual_cost_bps"), ".1f") + " bps", False, 30),
        ("Avg Daily Turnover",           f(costs.get("avg_daily_turnover"), pct=True),  False, 0.15),
        ("Total Costs Paid",             f(costs.get("total_costs_usd"), dollar=True),  False, 0),
    ]

    n = len(rows)
    fig, ax = plt.subplots(figsize=(9, n * 0.38 + 1.2))
    ax.axis("off")

    table_data = [[r[0], r[1]] for r in rows]
    col_labels = ["Metric", "Value"]

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width([0, 1])

    # Header styling
    for j in range(2):
        cell = tbl[0, j]
        cell.set_facecolor("#1976D2")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_height(0.055)

    # Row colouring
    for i, (label, value, good_positive, threshold) in enumerate(rows):
        row_idx = i + 1
        # Separator rows
        if label == "":
            for j in range(2):
                tbl[row_idx, j].set_facecolor("#e0e0e0")
                tbl[row_idx, j].set_height(0.008)
            continue

        # Alternating light background
        bg = "#f5f5f5" if i % 2 == 0 else "white"
        tbl[row_idx, 0].set_facecolor(bg)

        if good_positive is None or value == "N/A":
            tbl[row_idx, 1].set_facecolor(bg)
            continue

        try:
            raw_str = value.replace("$", "").replace(",", "").replace("%", "").replace("+", "").replace(" bps", "")
            v = float(raw_str)
            is_good = (v > threshold) if good_positive else (v < threshold)
            tbl[row_idx, 1].set_facecolor("#c8e6c9" if is_good else "#ffcdd2")
        except Exception:
            tbl[row_idx, 1].set_facecolor(bg)

    meta = m.get("meta", {})
    title = (
        f"Cross-Sectional Tech Stat-Arb  |  "
        f"{meta.get('start_date','2015-01-01')} → {meta.get('end_date','present')}  |  "
        f"$100k NAV"
    )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    path = os.path.join(RESULTS_DIR, "00_summary_table.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    metrics = _load_metrics()
    print_tearsheet(metrics)
    table_path = save_summary_table(metrics)
    print(f"  Summary table saved → {table_path}\n")
