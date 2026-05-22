"""
portfolio.py — Dollar-neutral, factor-neutral portfolio construction for OU stat-arb.

CONSTRUCTION LOGIC (per rebalance):
  1. Raw weights: w_i = −z_i / sigma_eq_i  (mean-reverting bet sized by OU vol)
     Negative z (asset below mean) → positive weight (long).
     Positive z (asset above mean) → negative weight (short).
  2. Zero-out non-tradeable assets (failed half-life filter or no OU params).
  3. Dollar-neutral: w ← w − mean(w)
  4. Factor-neutral projection: remove common-factor component from weights.
       w_neutral = w − B·(B^T·B)^{−1}·B^T·w
     where B (N×k) is the PCA factor loading matrix from FactorModel.fit().
  5. Scale to unit gross exposure: w ← w / sum(|w|)
     Then multiply by target notional:  dollar_weights = w × TARGET_NOTIONAL
  6. Single-position cap: clip each position to MAX_POSITION_FRAC × notional.

OUTPUTS:
  build_weights() returns a pd.Series of signed dollar positions (long+, short−).
"""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_NOTIONAL   = 1_000_000   # $1M portfolio
MAX_POSITION_FRAC = 0.10        # no single stock > 10% of notional


class PortfolioConstructor:
    """
    Convert OU z-scores into dollar-neutral, factor-neutral dollar positions.

    Parameters
    ----------
    target_notional   : float  Total gross notional (sum of |positions|).
    max_position_frac : float  Max fraction of notional per stock.
    """

    def __init__(
        self,
        target_notional:   float = TARGET_NOTIONAL,
        max_position_frac: float = MAX_POSITION_FRAC,
    ):
        self.target_notional   = target_notional
        self.max_position_frac = max_position_frac

    # ── Public API ─────────────────────────────────────────────────────────

    def build_weights(
        self,
        z_scores:  pd.Series,
        ou_params: dict,
        B:         np.ndarray,
        tickers:   List[str],
    ) -> pd.Series:
        """
        Build one day's dollar positions.

        Parameters
        ----------
        z_scores  : pd.Series   Current z-scores indexed by ticker (NaN = non-tradeable).
        ou_params : dict        OU parameters from SignalGenerator.fit(); keys are tickers.
        B         : np.ndarray  Factor loading matrix (N, k) from FactorModel.fit().
        tickers   : list[str]   Ordered list of all tickers (matches rows of B).

        Returns
        -------
        pd.Series  Signed dollar positions indexed by ticker.
                   Positive = long, Negative = short, 0 = no position.
        """
        n = len(tickers)
        w = np.zeros(n)

        # Step 1: raw OU weights for tradeable assets
        for i, t in enumerate(tickers):
            if t not in ou_params or t not in z_scores.index:
                continue
            z       = z_scores.get(t, np.nan)
            sig_eq  = ou_params[t]["sigma_eq"]
            if np.isnan(z) or sig_eq < 1e-10:
                continue
            w[i] = -z / sig_eq

        # Step 2: zero non-tradeable (already zero from initialization)

        if np.all(w == 0):
            return pd.Series(0.0, index=tickers)

        # Step 3: dollar-neutral
        w = w - w.mean()

        # Step 4: factor-neutral projection (only if B has columns)
        if B is not None and B.shape[1] > 0:
            w = self._factor_neutral(w, B)

        # Step 5: scale to unit gross exposure, then to target notional
        gross = np.abs(w).sum()
        if gross < 1e-10:
            return pd.Series(0.0, index=tickers)
        w = w / gross * self.target_notional

        # Step 6: position cap
        cap = self.max_position_frac * self.target_notional
        w   = np.clip(w, -cap, cap)

        return pd.Series(w, index=tickers)

    # ── Factor-Neutral Projection ──────────────────────────────────────────

    def _factor_neutral(self, w: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Project w onto the null-space of B to remove factor exposure.

        w_neutral = w − B·(B^T·B)^{−1}·B^T·w

        Parameters
        ----------
        w : np.ndarray  (N,) raw weights.
        B : np.ndarray  (N, k) factor loading matrix.

        Returns
        -------
        np.ndarray  (N,) factor-neutral weights.
        """
        try:
            BtB_inv = np.linalg.pinv(B.T @ B)          # (k, k)
            proj    = B @ BtB_inv @ B.T @ w             # (N,)
            return w - proj
        except np.linalg.LinAlgError:
            logger.warning("Factor-neutral projection failed (singular B^T B) — skipping.")
            return w
