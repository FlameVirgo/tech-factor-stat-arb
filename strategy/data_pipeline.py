"""
data_pipeline.py — Download, cache, clean, and prepare OHLCV data for the
                    cross-sectional tech stat-arb system.

WHAT THIS MODULE DOES:
    1. Batch-downloads OHLCV for all 50 universe stocks + SPY from Yahoo Finance.
    2. Caches each ticker as a Parquet file under data/raw/ — re-downloads only
       when the cache is stale (> CONFIG["cache_staleness_days"] old).
    3. Aligns all tickers to a shared trading calendar, dropping dates where
       more than 10% of the universe has missing data.
    4. Forward-fills short gaps (≤ 2 days) to handle occasional illiquid days.
    5. Drops individual tickers with more than 5% missing data in any rolling year.
    6. Computes daily log returns: r_t = log(P_t / P_{t-1}).
    7. Approximates historical market cap as shares_outstanding × price.
    8. Logs all data-quality issues to results/logs/data_quality.log.

OUTPUT (from .run()):
    dict with keys:
        "prices"       pd.DataFrame  (dates × tickers)  adjusted close prices
        "opens"        pd.DataFrame  (dates × tickers)  adjusted open prices
        "volumes"      pd.DataFrame  (dates × tickers)  trading volume
        "market_caps"  pd.DataFrame  (dates × tickers)  approx historical market cap ($)
        "log_returns"  pd.DataFrame  (dates × tickers)  daily log returns
        "spy_returns"  pd.Series     (dates,)            SPY log returns (benchmark)
        "universe"     list[str]     tickers that passed all quality filters
"""

import logging
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from config import CONFIG

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Manages the full data lifecycle: download → cache → clean → transform.

    Parameters
    ----------
    config : dict   CONFIG from config.py.
    """

    def __init__(self, config: dict = CONFIG):
        self.cfg = config
        self._setup_dirs()
        self._setup_quality_logger()

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Execute the full data pipeline.

        Returns
        -------
        dict   See module docstring for key descriptions.
        """
        logger.info("=== Data Pipeline ===")

        # 1. Load raw OHLCV for every ticker (universe + benchmark)
        all_tickers = self.cfg["universe"] + [self.cfg["benchmark"]]
        raw = self._load_all(all_tickers)

        # 2. Extract benchmark (SPY) before universe alignment
        spy_raw = raw.pop(self.cfg["benchmark"], None)
        if spy_raw is None:
            raise RuntimeError("SPY data missing — cannot build factor model.")

        # 3. Align universe to a shared trading calendar
        closes, opens, volumes = self._align_and_stack(raw)

        # 4. Drop dates where >10% of stocks are missing
        closes, opens, volumes = self._drop_sparse_dates(closes, opens, volumes)

        # 5. Forward-fill short gaps within each stock (≤ 2 trading days)
        closes  = closes.ffill(limit=self.cfg["max_fwd_fill_days"])
        opens   = opens.ffill(limit=self.cfg["max_fwd_fill_days"])
        volumes = volumes.ffill(limit=self.cfg["max_fwd_fill_days"])

        # 6. Drop stocks with >5% missing in any rolling year
        closes, opens, volumes, survivors = self._drop_sparse_stocks(
            closes, opens, volumes
        )

        # 7. Log returns  (log because additive across time; handles compounding)
        log_returns = np.log(closes / closes.shift(1))

        # 8. Approximate historical market cap: shares_outstanding × price
        market_caps = self._compute_market_caps(closes, survivors)

        # 9. SPY returns aligned to same calendar
        spy_close = spy_raw["Close"].reindex(closes.index).ffill(limit=2)
        spy_returns = np.log(spy_close / spy_close.shift(1)).rename("SPY")

        logger.info(
            f"Pipeline complete: {len(survivors)} stocks × {len(closes)} days  "
            f"({closes.index[0].date()} → {closes.index[-1].date()})"
        )

        return {
            "prices":      closes,
            "opens":       opens,
            "volumes":     volumes,
            "market_caps": market_caps,
            "log_returns": log_returns,
            "spy_returns": spy_returns,
            "universe":    survivors,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # DOWNLOAD & CACHE
    # ─────────────────────────────────────────────────────────────────────────

    def _load_all(self, tickers: list) -> dict:
        """
        Load OHLCV for every ticker; use local Parquet cache when fresh.

        Parameters
        ----------
        tickers : list[str]   All tickers to load (universe + SPY).

        Returns
        -------
        dict[str, pd.DataFrame]   Raw OHLCV per ticker.
        """
        raw = {}
        for ticker in tqdm(tickers, desc="Loading tickers", unit="ticker"):
            try:
                raw[ticker] = self._load_one(ticker)
            except Exception as exc:
                self._qlog(f"[{ticker}] Failed to load: {exc}")
                logger.warning(f"[{ticker}] Skipping due to load failure: {exc}")
        return raw

    def _load_one(self, ticker: str) -> pd.DataFrame:
        """
        Load one ticker from cache; re-download if stale or missing.

        Cache freshness: if the last cached date is older than
        CONFIG["cache_staleness_days"] calendar days, re-download.

        Parameters
        ----------
        ticker : str   Yahoo Finance ticker symbol.

        Returns
        -------
        pd.DataFrame   OHLCV with DatetimeIndex, columns Open/High/Low/Close/Volume.
        """
        cache_path = os.path.join(self.cfg["raw_data_dir"], f"{ticker}.parquet")

        if os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
            staleness = (datetime.today().date() - df.index[-1].date()).days
            if staleness <= self.cfg["cache_staleness_days"]:
                return df
            logger.info(f"[{ticker}] Cache {staleness}d old — re-downloading.")

        return self._download(ticker, cache_path)

    def _download(self, ticker: str, cache_path: str) -> pd.DataFrame:
        """
        Download OHLCV from Yahoo Finance with auto-adjustment and cache it.

        auto_adjust=True back-adjusts prices for stock splits and dividends,
        ensuring the return series has no artificial jumps at corporate-action dates.

        Parameters
        ----------
        ticker     : str   Ticker symbol.
        cache_path : str   Path to write Parquet cache.

        Returns
        -------
        pd.DataFrame   Adjusted OHLCV.
        """
        t = yf.Ticker(ticker)
        df = t.history(
            start=self.cfg["start_date"],
            end=self.cfg["end_date"],
            auto_adjust=True,
            actions=False,
        )

        if df.empty:
            raise ValueError(f"No data returned for {ticker}.")

        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.sort_index(inplace=True)

        # Keep only standard OHLCV columns
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].copy()
        df.to_parquet(cache_path)
        logger.debug(f"[{ticker}] Cached {len(df)} rows → {cache_path}")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # ALIGNMENT & CLEANING
    # ─────────────────────────────────────────────────────────────────────────

    def _align_and_stack(self, raw: dict) -> tuple:
        """
        Stack individual OHLCV DataFrames into wide matrices (dates × tickers).

        Only dates that appear in at least ONE ticker are included initially;
        sparse-date filtering happens next.

        Parameters
        ----------
        raw : dict[str, pd.DataFrame]   Per-ticker OHLCV.

        Returns
        -------
        (closes, opens, volumes) : three pd.DataFrames (dates × tickers)
        """
        closes  = pd.DataFrame({t: raw[t]["Close"]  for t in raw if "Close"  in raw[t].columns})
        opens   = pd.DataFrame({t: raw[t]["Open"]   for t in raw if "Open"   in raw[t].columns})
        volumes = pd.DataFrame({t: raw[t]["Volume"] for t in raw if "Volume" in raw[t].columns})

        # Sort chronologically
        closes.sort_index(inplace=True)
        opens.sort_index(inplace=True)
        volumes.sort_index(inplace=True)

        logger.info(f"Stacked {len(closes.columns)} tickers × {len(closes)} raw dates.")
        return closes, opens, volumes

    def _drop_sparse_dates(
        self, closes: pd.DataFrame, opens: pd.DataFrame, volumes: pd.DataFrame
    ) -> tuple:
        """
        Remove dates where more than CONFIG["max_missing_date_pct"] of stocks
        have missing (NaN) close prices.

        Rationale: a date where 10%+ of stocks are missing likely represents a
        partial trading day, data feed failure, or extreme market closure.
        Including it would distort cross-sectional statistics.

        Parameters
        ----------
        closes, opens, volumes : pd.DataFrames (dates × tickers)

        Returns
        -------
        Filtered (closes, opens, volumes) with sparse dates removed.
        """
        threshold = self.cfg["max_missing_date_pct"]
        missing_frac = closes.isna().mean(axis=1)
        good_dates = missing_frac[missing_frac <= threshold].index

        n_dropped = len(closes) - len(good_dates)
        if n_dropped > 0:
            self._qlog(f"Dropped {n_dropped} dates with >{threshold:.0%} missing stocks.")
            logger.warning(f"Dropped {n_dropped} sparse dates.")

        return closes.loc[good_dates], opens.loc[good_dates], volumes.loc[good_dates]

    def _drop_sparse_stocks(
        self, closes: pd.DataFrame, opens: pd.DataFrame, volumes: pd.DataFrame
    ) -> tuple:
        """
        Remove stocks with >5% missing data in ANY rolling 252-day window.

        A stock that repeatedly goes missing in a single year is likely illiquid,
        halted, or has data-feed problems — it will distort factor computations.

        Parameters
        ----------
        closes, opens, volumes : pd.DataFrames

        Returns
        -------
        (closes, opens, volumes, survivors) where survivors is list[str] of
        tickers that passed the quality filter.
        """
        threshold = self.cfg["max_missing_stock_pct"]
        window    = self.cfg["trading_days_per_year"]

        # Rolling missing fraction per stock: max over all windows
        rolling_missing = closes.isna().rolling(window, min_periods=window // 2).mean()
        max_missing_per_stock = rolling_missing.max()

        survivors = max_missing_per_stock[max_missing_per_stock <= threshold].index.tolist()
        dropped   = [t for t in closes.columns if t not in survivors]

        if dropped:
            self._qlog(f"Dropped stocks with >5% rolling missing: {dropped}")
            logger.warning(f"Dropped {len(dropped)} sparse stocks: {dropped}")

        return (
            closes[survivors],
            opens[survivors],
            volumes[survivors],
            survivors,
        )

    def _compute_market_caps(
        self, prices: pd.DataFrame, tickers: list
    ) -> pd.DataFrame:
        """
        Approximate historical market capitalisation as shares_outstanding × price.

        APPROXIMATION:
            We use the CURRENT shares outstanding from Yahoo Finance and scale by
            historical price.  This is equivalent to assuming shares outstanding
            has been constant over the period.  For large-cap tech stocks that
            have done buybacks this introduces a small upward bias in historical
            size — acceptable for a size-factor proxy.

        Parameters
        ----------
        prices  : pd.DataFrame   Historical adjusted close prices.
        tickers : list[str]      Tickers to compute market cap for.

        Returns
        -------
        pd.DataFrame   Approximate market cap in USD (dates × tickers).
        """
        shares = {}
        for ticker in tqdm(tickers, desc="Fetching shares outstanding", unit="ticker"):
            try:
                info = yf.Ticker(ticker).fast_info
                # fast_info.shares is current shares outstanding
                sh = getattr(info, "shares", None)
                if sh and sh > 0:
                    shares[ticker] = sh
                else:
                    # Fallback: derive from current market cap / current price
                    mktcap = getattr(info, "market_cap", None)
                    price  = getattr(info, "last_price", None)
                    if mktcap and price and price > 0:
                        shares[ticker] = mktcap / price
            except Exception as exc:
                self._qlog(f"[{ticker}] Could not fetch shares outstanding: {exc}")

        mkt_caps = pd.DataFrame(index=prices.index, columns=tickers, dtype=float)
        for ticker in tickers:
            if ticker in shares:
                mkt_caps[ticker] = prices[ticker] * shares[ticker]
            else:
                # Last resort: use price as a size proxy (proportional to mktcap
                # assuming shares outstanding is roughly constant cross-sectionally)
                self._qlog(f"[{ticker}] Using price as mktcap proxy (shares unknown).")
                mkt_caps[ticker] = prices[ticker] * 1e8   # arbitrary scale

        return mkt_caps.astype(float)

    # ─────────────────────────────────────────────────────────────────────────
    # SETUP HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        for key in ["raw_data_dir", "processed_data_dir", "factors_dir",
                    "results_dir", "plots_dir", "logs_dir", "tearsheets_dir"]:
            os.makedirs(self.cfg[key], exist_ok=True)

    def _setup_quality_logger(self) -> None:
        """Set up a dedicated file logger for data-quality issues."""
        log_path = os.path.join(self.cfg["logs_dir"], "data_quality.log")
        self._qlogger = logging.getLogger("data_quality")
        if not self._qlogger.handlers:
            fh = logging.FileHandler(log_path, mode="w")
            fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M"))
            self._qlogger.addHandler(fh)
            self._qlogger.setLevel(logging.INFO)
            self._qlogger.propagate = False

    def _qlog(self, msg: str) -> None:
        """Write a quality-issue message to the dedicated data-quality log file."""
        self._qlogger.info(msg)
