"""
data.py — Download, clean, and cache price data for the OU stat-arb system.

PIPELINE:
  1. Batch-download 5+ years of daily adjusted close prices via yfinance.
  2. Drop tickers with > 5% missing observations.
  3. Forward-fill then back-fill remaining gaps (max 3 consecutive days).
  4. Compute daily log returns: r_t = log(P_t / P_{t-1}).
  5. Cache prices and log-returns as Parquet files under data/.
  6. Print a data-quality report.
"""

import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

TICKERS: list[str] = [
    "NVDA", "AAPL", "MSFT", "GOOGL", "META", "AVGO", "AMD", "ORCL", "QCOM", "TXN",
    "NOW",  "INTU", "AMAT", "LRCX",  "SNPS", "CDNS", "KLAC", "ADI",  "MRVL", "CRM",
    "PANW", "CRWD", "ADBE", "NFLX",  "SHOP", "UBER", "INTC", "HPQ",  "DELL", "FTNT",
    "NET",  "DDOG", "WDAY", "TTD",   "ZS",   "TEAM", "HUBS", "MDB",  "OKTA", "PATH",
    "SNOW", "RBLX", "DUOL", "MNDY",  "ANSS", "PTC",  "EPAM", "CTSH", "PSTG", "ARM",
]

START_DATE       = "2019-01-01"   # extra history for the 252-day rolling burn-in
END_DATE         = None           # None → today
DATA_DIR         = "data"
MAX_MISSING_PCT  = 0.05           # drop ticker if > 5% of dates are NaN
MAX_FFILL_DAYS   = 3              # forward-fill / back-fill at most this many days

PRICES_FILE      = os.path.join(DATA_DIR, "prices.parquet")
LOG_RETURNS_FILE = os.path.join(DATA_DIR, "log_returns.parquet")


# ── DataManager ───────────────────────────────────────────────────────────────

class DataManager:
    """
    Handles the full data lifecycle: download → clean → parquet cache → load.
    """

    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────

    def download_and_clean(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download raw prices, apply quality filters, save parquet, return
        (prices, log_returns).
        """
        logger.info("=== Data Download ===")
        raw = self._download(TICKERS)
        prices = self._clean(raw)
        log_returns = np.log(prices / prices.shift(1))
        self._save(prices, log_returns)
        self._quality_report(prices, log_returns)
        return prices, log_returns

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load cached parquet files. Call download_and_clean() first."""
        if not os.path.exists(PRICES_FILE):
            raise FileNotFoundError(
                f"No cached data at {PRICES_FILE}. Run without --skip-download first."
            )
        prices      = pd.read_parquet(PRICES_FILE)
        log_returns = pd.read_parquet(LOG_RETURNS_FILE)
        logger.info(
            f"Loaded {len(prices.columns)} tickers × {len(prices)} days  "
            f"({prices.index[0].date()} → {prices.index[-1].date()})"
        )
        return prices, log_returns

    # ── Download ───────────────────────────────────────────────────────────

    def _download(self, tickers: list[str]) -> pd.DataFrame:
        """Batch-download adjusted close prices via yfinance."""
        logger.info(f"Downloading {len(tickers)} tickers from {START_DATE} …")
        raw = yf.download(
            tickers,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=True,
            progress=True,
            threads=True,
        )
        # yfinance returns MultiIndex columns when > 1 ticker
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw["Close"]
        else:
            raw.columns = [tickers[0]]

        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        raw.sort_index(inplace=True)
        logger.info(f"Raw download: {raw.shape[1]} tickers × {raw.shape[0]} dates.")
        return raw

    # ── Cleaning ───────────────────────────────────────────────────────────

    def _clean(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Apply quality filters and fill gaps."""
        # 1. Drop tickers with > MAX_MISSING_PCT NaN
        missing_frac = raw.isna().mean()
        good = missing_frac[missing_frac <= MAX_MISSING_PCT].index.tolist()
        dropped = [t for t in raw.columns if t not in good]
        if dropped:
            logger.warning(f"Dropped {len(dropped)} tickers (>{MAX_MISSING_PCT:.0%} missing): {dropped}")
        prices = raw[good].copy()

        # 2. Forward-fill then back-fill (max MAX_FFILL_DAYS consecutive days)
        prices = prices.ffill(limit=MAX_FFILL_DAYS).bfill(limit=MAX_FFILL_DAYS)

        # 3. Drop any remaining rows where ALL assets are NaN (e.g. market holidays)
        prices = prices.dropna(how="all")

        # 4. Drop any residual columns with NaN after filling
        prices = prices.dropna(axis=1)

        logger.info(f"After cleaning: {prices.shape[1]} tickers × {prices.shape[0]} dates.")
        return prices

    # ── Save / Load ────────────────────────────────────────────────────────

    def _save(self, prices: pd.DataFrame, log_returns: pd.DataFrame) -> None:
        prices.to_parquet(PRICES_FILE)
        log_returns.to_parquet(LOG_RETURNS_FILE)
        logger.info(f"Cached prices → {PRICES_FILE}")
        logger.info(f"Cached log-returns → {LOG_RETURNS_FILE}")

    # ── Quality Report ─────────────────────────────────────────────────────

    def _quality_report(self, prices: pd.DataFrame, log_returns: pd.DataFrame) -> None:
        """Print a data-quality summary to stdout."""
        n_tickers = prices.shape[1]
        n_days    = prices.shape[0]
        date_start = prices.index[0].date()
        date_end   = prices.index[-1].date()

        print()
        print("=" * 60)
        print("  DATA QUALITY REPORT")
        print("=" * 60)
        print(f"  Tickers retained : {n_tickers}  (of {len(TICKERS)} requested)")
        print(f"  Date range       : {date_start}  →  {date_end}  ({n_days} days)")
        print(f"  Missing after fill: {prices.isna().sum().sum()} cells")
        print()
        print(f"  {'Ticker':<8} {'Missing%':>10} {'Start':>12} {'End':>12}")
        print("  " + "-" * 46)
        missing_pct = prices.isna().mean().sort_values(ascending=False)
        for ticker, pct in missing_pct.items():
            first_valid = prices[ticker].first_valid_index()
            last_valid  = prices[ticker].last_valid_index()
            print(f"  {ticker:<8} {pct:>10.2%} {str(first_valid.date()):>12} {str(last_valid.date()):>12}")
        print("=" * 60)
        print()
