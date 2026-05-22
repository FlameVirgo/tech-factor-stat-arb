"""
main.py — Full pipeline orchestrator for the cross-sectional tech stat-arb strategy.

PIPELINE (9 steps):
    1. Data download & caching       (DataPipeline)
    2. Factor model computation      (FactorModel)
    3. Signal generation             (SignalGenerator)
    4. Portfolio construction        (PortfolioConstructor)
    5. Backtesting                   (Backtester)
    6. Performance analytics         (PerformanceAnalytics)
    7. Visualizations                (Visualizer)
    8. Metrics export                results/metrics.json
    9. Tearsheet console print       results/summary.py logic

Run with:
    python main.py

Outputs:
    results/metrics.json               — full metrics dict for downstream scripts
    results/00_summary_table.png       — colour-coded metrics table
    results/01_cumulative_pnl.png      — 10 research charts
    ...
    results/10_trade_pnl_histogram.png
    results/logs/                      — data quality and run logs
"""

import datetime
import json
import logging
import os
import sys

# ── Logging setup (before any imports that use loggers) ─────────────────────
os.makedirs("results/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/logs/run.log", mode="w"),
    ],
)
logger = logging.getLogger("main")

# ── Strategy imports ─────────────────────────────────────────────────────────
from config import CONFIG
from strategy.data_pipeline   import DataPipeline
from strategy.factor_model    import FactorModel
from strategy.signal          import SignalGenerator
from strategy.portfolio       import PortfolioConstructor
from strategy.backtest        import Backtester
from strategy.performance     import PerformanceAnalytics
from strategy.visualizations  import Visualizer
from results.summary          import print_tearsheet, save_summary_table


# ─────────────────────────────────────────────────────────────────────────────
# JSON serialisation helper
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize(obj):
    """
    Recursively convert any non-JSON-serialisable value to a safe equivalent.

    Handles:
      - float NaN / ±inf  → None
      - numpy scalars     → Python native int/float
      - datetime.date / datetime.datetime → ISO string
      - pandas Series     → drop (too large / not useful in JSON)
      - pandas DataFrame  → drop
      - dict / list       → recurse
    """
    import datetime as dt
    import pandas as pd
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return None   # visualisation artefacts — not useful in JSON
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    if isinstance(obj, (dt.date, dt.datetime)):
        return str(obj)
    if isinstance(obj, bool):
        return obj
    if hasattr(obj, "item"):          # numpy scalar → Python native
        val = obj.item()
        if isinstance(val, float) and (val != val or val in (float("inf"), float("-inf"))):
            return None
        return val
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("  CROSS-SECTIONAL TECH STAT-ARB  —  FULL PIPELINE")
    logger.info("=" * 60)

    # ── STEP 1: Data ─────────────────────────────────────────────────────────
    logger.info("[1/9] Downloading and caching market data …")
    pipeline = DataPipeline(CONFIG)
    data     = pipeline.run()
    logger.info(
        f"  Universe: {len(data['universe'])} stocks  |  "
        f"{data['log_returns'].index[0].date()} → {data['log_returns'].index[-1].date()}"
    )

    # ── STEP 2: Factor Model ─────────────────────────────────────────────────
    logger.info("[2/9] Computing five-factor risk model …")
    factor_model = FactorModel(data, CONFIG)
    factors      = factor_model.compute_all_factors()

    # ── STEP 3: Signal Generation ────────────────────────────────────────────
    logger.info("[3/9] Generating Fama-MacBeth idiosyncratic return signal …")
    signal_gen   = SignalGenerator(data, factors, CONFIG)
    signals, factor_returns, residuals = signal_gen.generate()

    # IC diagnostic
    ic_stats = signal_gen.compute_ic(signals, data["log_returns"])
    logger.info(
        f"  IC mean={ic_stats['ic_mean']:.4f}  "
        f"ICIR={ic_stats['icir']:.3f}  "
        f"t-stat={ic_stats['ic_tstat']:.2f}"
    )

    # ── STEP 4: Portfolio Construction ───────────────────────────────────────
    logger.info("[4/9] Building weekly dollar-neutral long/short portfolio …")
    port_constructor = PortfolioConstructor(signals, factors, data["prices"], CONFIG)
    positions, trades, factor_exp_ts = port_constructor.build_positions()
    logger.info(
        f"  Rebalances: {(trades.abs().sum(axis=1) > 0).sum()}  |  "
        f"Avg daily turnover: {port_constructor.compute_turnover(trades).mean():.1%}"
    )

    # ── STEP 5: Backtesting ──────────────────────────────────────────────────
    logger.info("[5/9] Running event-driven backtest …")
    backtester = Backtester(positions, trades, data, CONFIG)
    daily_pnl, gross_pnl, cost_series, book_stats = backtester.run()

    # ── STEP 6: Performance Analytics ───────────────────────────────────────
    logger.info("[6/9] Computing performance metrics …")
    analytics = PerformanceAnalytics(
        net_pnl       = daily_pnl,
        gross_pnl     = gross_pnl,
        cost_series   = cost_series,
        positions     = positions,
        trades        = trades,
        data          = data,
        factors       = factors,
        factor_returns= factor_returns,
        signals       = signals,
        ic_stats      = ic_stats,
        factor_exp_ts = factor_exp_ts,
        config        = CONFIG,
    )
    metrics = analytics.compute_all()

    # Attach metadata for the tearsheet
    metrics["meta"] = {
        "start_date":    str(daily_pnl.index[0].date()),
        "end_date":      str(daily_pnl.index[-1].date()),
        "n_stocks":      len(data["universe"]),
        "generated_at":  str(datetime.datetime.now())[:19],
    }

    # ── STEP 7: Visualizations ───────────────────────────────────────────────
    logger.info("[7/9] Generating 10 research charts …")
    viz = Visualizer(
        daily_pnl     = daily_pnl,
        gross_pnl     = gross_pnl,
        cost_series   = cost_series,
        book_stats    = book_stats,
        factor_exp_ts = factor_exp_ts,
        signals       = signals,
        residuals     = residuals,
        trades        = trades,
        metrics       = metrics,
        spy_returns   = data["spy_returns"],
        config        = CONFIG,
    )
    chart_paths = viz.plot_all()
    logger.info(f"  {len(chart_paths)} charts saved to results/")

    # ── STEP 8: Export metrics.json ──────────────────────────────────────────
    logger.info("[8/9] Exporting metrics to results/metrics.json …")
    os.makedirs("results", exist_ok=True)
    metrics_path = "results/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(_sanitize(metrics), f, indent=2)
    logger.info(f"  Saved: {metrics_path}")

    # ── STEP 9: Print tearsheet ──────────────────────────────────────────────
    logger.info("[9/9] Printing performance tearsheet …")
    print_tearsheet(metrics)
    table_path = save_summary_table(metrics)
    logger.info(f"  Summary table → {table_path}")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
