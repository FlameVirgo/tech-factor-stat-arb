"""
factors.py — Rolling PCA factor model with Ledoit-Wolf shrinkage.

ALGORITHM (per rebalance window):
  1. Take trailing ROLLING_WINDOW-day log-return matrix R  (T × N).
  2. Standardize each asset's return series to zero mean, unit variance.
  3. Compute sample covariance Σ  (N × N) on the standardized matrix.
  4. Apply Ledoit-Wolf analytical shrinkage.
  5. Eigendecompose: Σ_shrunk = Q Λ Q^T  (ascending eigenvalues from eigh).
  6. Retain top-k eigenvectors explaining ≥ VARIANCE_THRESHOLD of total variance.
  7. Factor returns: F = Q_k^T · R^T          shape (k, T)   [R^T is N×T]
  8. OLS betas:     B = R^T · F^T · (F·F^T)⁻¹  shape (N, k)
  9. Residuals:     E = R^T − B · F            shape (N, T)  → stored as (T, N)
     … then un-standardized back to log-return units.

Between rebalances, project_new_return() applies the LAST fitted model to
a single new day's returns to obtain same-day residuals for stop-loss checking.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

VARIANCE_THRESHOLD = 0.55   # retain factors explaining ≥ 55 % of total variance
ROLLING_WINDOW     = 252    # look-back window in trading days


class FactorModel:
    """
    Ledoit-Wolf PCA factor model.

    Call fit(returns_window) where returns_window is a (T, N) DataFrame.
    """

    # ── Public API ─────────────────────────────────────────────────────────

    def fit(self, returns_window: pd.DataFrame) -> dict:
        """
        Fit the factor model on a trailing window of log returns.

        Parameters
        ----------
        returns_window : pd.DataFrame  shape (T, N) — T days, N assets.

        Returns
        -------
        dict with keys:
            B            np.ndarray (N, k)    factor loadings in standardised space
            F            np.ndarray (k, T)    factor return series (standardised)
            E            pd.DataFrame (T, N)  residuals in log-return space
            Q_k          np.ndarray (N, k)    top-k eigenvectors of Σ_shrunk
            k            int                  number of factors retained
            explained_var float               cumulative variance fraction
            means        pd.Series            per-asset return means
            stds         pd.Series            per-asset return stds
            tickers      list[str]
        """
        returns_window = returns_window.dropna()
        T, N = returns_window.shape
        tickers = list(returns_window.columns)

        # ── Step 2: Standardise ────────────────────────────────────────────
        means = returns_window.mean()
        stds  = returns_window.std().replace(0.0, 1e-8)
        R_std = (returns_window - means) / stds          # (T, N)

        # ── Step 3-4: Ledoit-Wolf shrinkage on standardised covariance ─────
        lw = LedoitWolf(assume_centered=False)
        lw.fit(R_std.values)
        Sigma_shrunk = lw.covariance_                    # (N, N)

        # ── Step 5: Eigendecomposition (eigh returns ascending order) ──────
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma_shrunk)
        idx        = np.argsort(eigenvalues)[::-1]       # sort descending
        eigenvalues  = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]              # (N, N)

        # ── Step 6: Retain top-k explaining ≥ VARIANCE_THRESHOLD ───────────
        total_var = eigenvalues.sum()
        cum_var   = np.cumsum(eigenvalues) / total_var
        k = int(np.searchsorted(cum_var, VARIANCE_THRESHOLD) + 1)
        k = max(1, min(k, N - 1))

        Q_k          = eigenvectors[:, :k]               # (N, k)
        explained_var = float(cum_var[k - 1])

        logger.debug(f"  k={k} factors  |  explained={explained_var:.1%}")

        # ── Step 7: Factor returns F = Q_k^T · R_std^T  ───────────────────
        R_std_T = R_std.values.T                         # (N, T)
        F = Q_k.T @ R_std_T                              # (k, T)

        # ── Step 8: OLS betas B = R_std^T · F^T · (F·F^T)⁻¹  ─────────────
        FF_inv = np.linalg.pinv(F @ F.T)                # (k, k)
        B      = R_std_T @ F.T @ FF_inv                 # (N, k)

        # ── Step 9: Residuals E = R_std^T − B · F  (in standardised space) ─
        E_std   = R_std_T - B @ F                       # (N, T)

        # Un-standardise back to log-return units
        E_logret = E_std * stds.values[:, np.newaxis]   # (N, T)

        E_df = pd.DataFrame(
            E_logret.T,
            index=returns_window.index,
            columns=tickers,
        )

        return {
            "B":            B,
            "F":            F,
            "E":            E_df,
            "Q_k":          Q_k,
            "k":            k,
            "explained_var": explained_var,
            "means":        means,
            "stds":         stds,
            "tickers":      tickers,
        }

    def project_new_return(
        self, r_new: pd.Series, fit_result: dict
    ) -> pd.Series:
        """
        Project a single new day's log-returns onto the last fitted factor model
        and return per-asset residuals (log-return scale).

        Used daily between rebalances for stop-loss checking — avoids
        refitting the full model every day.

        Parameters
        ----------
        r_new      : pd.Series  Today's log-returns (indexed by ticker).
        fit_result : dict       Output of fit().

        Returns
        -------
        pd.Series   Factor-model residuals for each asset (log-return scale).
        """
        tickers = fit_result["tickers"]
        means   = fit_result["means"]
        stds    = fit_result["stds"]
        Q_k     = fit_result["Q_k"]
        B       = fit_result["B"]

        r = r_new.reindex(tickers).fillna(0.0)

        # Standardise
        r_std = (r - means) / stds                       # (N,)

        # Factor return for this day: f = Q_k^T @ r_std
        f = Q_k.T @ r_std.values                        # (k,)

        # Fitted: r_fitted_std = B @ f
        r_fitted_std = B @ f                            # (N,)

        # Residual (standardised) → un-standardise
        e_std    = r_std.values - r_fitted_std          # (N,)
        e_logret = e_std * stds.values                  # (N,)

        return pd.Series(e_logret, index=tickers)
