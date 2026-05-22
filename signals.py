"""
signals.py — OU process fitting and z-score signal generation.

ALGORITHM (per rebalance window):
  For each asset i in the residual series E[:,i]:
  1. Fit AR(1) OLS regression:  Δe_t = a + b·e_{t-1} + ε_t
  2. Extract OU parameters:
       kappa    = −b / dt          (mean-reversion speed)
       mu       = −a / b           (long-run mean)
       sigma_eq = std(ε) / sqrt(2·kappa·dt)  (equilibrium std)
       half_life = log(2) / kappa  (in trading days)
  3. Filter: keep only assets with 5 ≤ half_life ≤ 40 days.
  4. Z-score: z_t = (e_t − mu) / sigma_eq
  5. Thresholds: entry |z| ≥ 2.0, exit |z| ≤ 0.5, stop-loss |z| ≥ 4.0.

Between rebalances, update_signals() applies the LAST fitted OU parameters
to new daily residuals to compute intra-week z-scores for stop-loss checks.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────

HALF_LIFE_MIN =  5   # trading days — too fast → noise
HALF_LIFE_MAX = 40   # trading days — too slow → no edge
ENTRY_Z       =  2.0
EXIT_Z        =  0.5
STOP_Z        =  4.0


class SignalGenerator:
    """
    Fit OU process parameters to factor-model residuals and compute z-scores.

    Call fit(E) once per rebalance window.
    Call zscore_single(e_new, ou_params) daily between rebalances.
    """

    # ── Public API ─────────────────────────────────────────────────────────

    def fit(self, E: pd.DataFrame) -> dict:
        """
        Fit OU processes to a residual matrix and return z-score signals.

        Parameters
        ----------
        E : pd.DataFrame  shape (T, N) — factor-model residuals in log-return space.

        Returns
        -------
        dict with keys:
            ou_params   dict[ticker → dict]  OU parameters for each tradeable asset
            z_scores    pd.DataFrame (T, N)  z-score time series (NaN where not tradeable)
            tradeable   list[str]            tickers passing the half-life filter
        """
        tickers = list(E.columns)
        ou_params: dict[str, dict] = {}
        z_scores = pd.DataFrame(np.nan, index=E.index, columns=tickers)

        for ticker in tickers:
            e = E[ticker].dropna().values.astype(float)
            if len(e) < 30:
                continue

            params = self._fit_ou(e)
            if params is None:
                continue

            hl = params["half_life"]
            if not (HALF_LIFE_MIN <= hl <= HALF_LIFE_MAX):
                logger.debug(
                    f"  {ticker}: half_life={hl:.1f}d — outside [{HALF_LIFE_MIN},{HALF_LIFE_MAX}], skipped"
                )
                continue

            ou_params[ticker] = params

            # Z-score the cumulative spread level using fitted mu and sigma_eq
            mu       = params["mu"]
            sigma_eq = params["sigma_eq"]
            if sigma_eq < 1e-10:
                continue

            e_cumsum = np.cumsum(E[ticker].values)
            z_scores[ticker] = (e_cumsum - mu) / sigma_eq
            ou_params[ticker]["last_level"] = float(e_cumsum[-1])

        tradeable = list(ou_params.keys())
        logger.debug(
            f"  Tradeable assets: {len(tradeable)}/{len(tickers)}  "
            f"(half-life filter {HALF_LIFE_MIN}–{HALF_LIFE_MAX}d)"
        )

        return {
            "ou_params":  ou_params,
            "z_scores":   z_scores,
            "tradeable":  tradeable,
        }

    def zscore_single(
        self, e_new: pd.Series, ou_params: dict, running_levels: dict
    ) -> pd.Series:
        """
        Compute z-scores for a single new day's residuals using stored OU params.

        Updates running_levels in-place by adding today's residual to the
        cumulative spread level carried from the last rebalance.

        Used daily between rebalances for stop-loss checking.

        Parameters
        ----------
        e_new          : pd.Series  New day's factor-model residuals (indexed by ticker).
        ou_params      : dict       Output of fit()["ou_params"].
        running_levels : dict       Mutable dict of {ticker: cumulative_level}; updated here.

        Returns
        -------
        pd.Series  Z-scores for tradeable tickers (NaN for non-tradeable).
        """
        z = {}
        for ticker, params in ou_params.items():
            e_val = e_new.get(ticker, np.nan)
            if np.isnan(e_val):
                z[ticker] = np.nan
                continue
            running_levels[ticker] = running_levels.get(ticker, 0.0) + e_val
            level    = running_levels[ticker]
            sigma_eq = params["sigma_eq"]
            mu       = params["mu"]
            z[ticker] = (level - mu) / sigma_eq if sigma_eq > 1e-10 else np.nan
        return pd.Series(z)

    # ── OU Parameter Estimation ────────────────────────────────────────────

    def _fit_ou(self, e: np.ndarray) -> Optional[dict]:
        """
        Fit OU parameters via AR(1) OLS on Δe_t = a + b·e_{t-1} + ε_t.

        Parameters
        ----------
        e : np.ndarray  1-D array of residuals (log-return units).

        Returns
        -------
        dict with keys: kappa, mu, sigma_eq, half_life
        or None if the fit is degenerate.
        """
        dt = 1.0  # 1 trading day

        # e is daily log-return residuals (changes); integrate to get spread level
        e_level = np.cumsum(e)      # length T
        delta_e = np.diff(e_level)  # = e[1:], length T-1
        e_lag   = e_level[:-1]      # e_level_{t-1}

        # OLS: [delta_e] = X @ [a, b]  where X = [1, e_lag]
        X = np.column_stack([np.ones_like(e_lag), e_lag])
        try:
            coeffs, residuals, rank, _ = np.linalg.lstsq(X, delta_e, rcond=None)
        except np.linalg.LinAlgError:
            return None

        a, b = coeffs

        # b must be negative for mean-reversion (kappa > 0)
        if b >= 0 or b <= -2:
            return None

        kappa = -b / dt
        if kappa < 1e-6:
            return None

        mu        = -a / b
        eps       = delta_e - X @ coeffs
        sigma_eps = np.std(eps, ddof=2)
        sigma_eq  = sigma_eps / np.sqrt(2.0 * kappa * dt)
        half_life = np.log(2.0) / kappa

        return {
            "kappa":     kappa,
            "mu":        mu,
            "sigma_eq":  sigma_eq,
            "half_life": half_life,
            "sigma_eps": sigma_eps,
        }
