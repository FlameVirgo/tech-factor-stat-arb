"""
main.py — CLI entrypoint for the OU stat-arb system.

Usage:
    python main.py                  # download fresh data, run full backtest
    python main.py --skip-download  # use cached parquet files in data/
    python main.py --notional 500000 --skip-download

Pipeline:
    1. DataManager   — download / load prices & log-returns
    2. Backtester    — walk-forward OU stat-arb simulation
    3. Reporter      — metrics, charts, tearsheet
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from data      import DataManager
from backtest  import Backtester
from report    import Reporter

# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("results/run.log", mode="w"),
        ],
    )

# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OU Stat-Arb Backtester — 50 US Tech Stocks"
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Load cached parquet files instead of downloading fresh data.",
    )
    p.add_argument(
        "--notional",
        type=float,
        default=1_000_000,
        help="Target gross notional in USD (default: 1,000,000).",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return p.parse_args()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    Path("results").mkdir(exist_ok=True)
    _setup_logging(args.verbose)
    logger = logging.getLogger("main")

    t0 = time.time()
    logger.info("=" * 60)
    logger.info("  OU STAT-ARB SYSTEM — START")
    logger.info("=" * 60)

    # ── 1. Data ───────────────────────────────────────────────────────────
    dm = DataManager()
    if args.skip_download:
        logger.info("Loading cached data …")
        prices, log_returns = dm.load()
    else:
        prices, log_returns = dm.download_and_clean()

    logger.info(
        f"Data loaded: {log_returns.shape[1]} tickers × {log_returns.shape[0]} days"
    )

    # ── 2. Backtest ───────────────────────────────────────────────────────
    logger.info("Starting walk-forward backtest …")
    bt = Backtester(
        log_returns     = log_returns,
        prices          = prices,
        target_notional = args.notional,
    )
    result = bt.run()
    elapsed_bt = time.time() - t0
    logger.info(f"Backtest complete in {elapsed_bt:.1f}s")

    # ── 3. Report ─────────────────────────────────────────────────────────
    logger.info("Generating performance report …")
    rep = Reporter(
        pnl_gross = result["pnl_gross"],
        pnl_net   = result["pnl_net"],
        costs     = result["costs"],
        trades    = result["trades"],
        positions = result["positions"],
        notional  = args.notional,
    )
    metrics = rep.run()

    elapsed = time.time() - t0
    logger.info(f"Done. Total elapsed: {elapsed:.1f}s")
    logger.info(
        f"Net Sharpe={metrics['sharpe_net']:.3f}  "
        f"Ann.Return={metrics['ann_return_net']:+.2%}  "
        f"MDD={metrics['max_drawdown']:.2%}"
    )

if __name__ == "__main__":
    main()
