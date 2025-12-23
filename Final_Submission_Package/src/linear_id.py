"""linear_id_fixed.py

A fair linear baseline for scalar or multivariate firing-rate time series.

Model:
    y[t+1] = A y[t] + B u[t] + c

- y can be shape (T,) or (T, ny)
- u can be shape (T,) or (T, nu)

Fitting is done by ridge regression. Optionally uses RidgeCV.
Optionally projects A to be stable by scaling if spectral radius exceeds max_rho.

Metrics:
- NMSE = MSE / Var(y_true)   (0 is perfect; 1 matches a mean predictor)
- R2   = 1 - NMSE

This file is intended to replace/augment src/linear_id.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

try:
    from sklearn.linear_model import Ridge, RidgeCV
except Exception as e:  # pragma: no cover
    Ridge = None
    RidgeCV = None


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {x.shape}")
    return x


def spectral_radius(A: np.ndarray) -> float:
    """Max absolute eigenvalue (spectral radius)."""
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got {A.shape}")
    if A.size == 1:
        return float(abs(A[0, 0]))
    eig = np.linalg.eigvals(A)
    return float(np.max(np.abs(eig)))


@dataclass
class FitDiagnostics:
    alpha: float
    rho: float
    mse_train: float
    nmse_train: float
    r2_train: float


class LinearSystemID:
    """Affine linear system ID with ridge regression + optional stability."""

    def __init__(
        self,
        *,
        dt_ms: float,
        enforce_stability: bool = True,
        max_rho: float = 0.95,
        ridge_alphas: Optional[Iterable[float]] = None,
        use_cv: bool = True,
    ) -> None:
        self.dt_ms = float(dt_ms)
        self.enforce_stability = bool(enforce_stability)
        self.max_rho = float(max_rho)
        self.use_cv = bool(use_cv)
        if ridge_alphas is None:
            # Wide-ish range; RidgeCV picks.
            ridge_alphas = np.logspace(-6, 3, 30)
        self.ridge_alphas = np.array(list(ridge_alphas), dtype=float)

        self.A: Optional[np.ndarray] = None
        self.B: Optional[np.ndarray] = None
        self.c: Optional[np.ndarray] = None
        self.diag: Optional[FitDiagnostics] = None

    def fit(self, y: np.ndarray, u: np.ndarray, *, burn_in: int = 0) -> FitDiagnostics:
        """Fit the model.

        burn_in: number of initial samples to ignore (transient).
        """
        y2 = _as_2d(np.asarray(y))
        u2 = _as_2d(np.asarray(u))

        if y2.shape[0] != u2.shape[0]:
            raise ValueError(f"y and u must have same length, got {y2.shape[0]} vs {u2.shape[0]}")
        if y2.shape[0] < 3:
            raise ValueError("Need at least 3 samples")

        T, ny = y2.shape
        _, nu = u2.shape

        start = int(burn_in)
        if start < 0 or start >= T - 2:
            raise ValueError(f"burn_in={burn_in} leaves too little data")

        # Regression: predict y[t+1] from [y[t], u[t], 1]
        X = np.concatenate(
            [y2[start : T - 1], u2[start : T - 1], np.ones((T - 1 - start, 1))],
            axis=1,
        )
        Y = y2[start + 1 : T]

        if Ridge is None:
            raise ImportError("scikit-learn is required for LinearSystemID (Ridge/RidgeCV).")

        if self.use_cv and RidgeCV is not None:
            model = RidgeCV(alphas=self.ridge_alphas, fit_intercept=False)
        else:
            model = Ridge(alpha=float(self.ridge_alphas[len(self.ridge_alphas) // 2]), fit_intercept=False)

        model.fit(X, Y)

        # coef_ shape: (ny, ny+nu+1)
        coef = np.asarray(model.coef_)
        A = coef[:, :ny]
        B = coef[:, ny : ny + nu]
        c = coef[:, -1]

        rho = spectral_radius(A)
        if self.enforce_stability and rho > self.max_rho and rho > 0:
            A = A * (self.max_rho / rho)
            rho = self.max_rho

        # Store
        self.A, self.B, self.c = A, B, c

        # Training diagnostics (teacher-forced)
        y_hat = self.predict_one_step(y2, u2)
        # Align: y_hat[t] predicts y[t]
        mse = float(np.mean((y2[1:] - y_hat[1:]) ** 2))
        nmse = self.nmse(y2[1:], y_hat[1:])
        r2 = 1.0 - nmse

        alpha = float(getattr(model, "alpha_", getattr(model, "alpha", np.nan)))
        self.diag = FitDiagnostics(alpha=alpha, rho=rho, mse_train=mse, nmse_train=nmse, r2_train=r2)
        return self.diag

    def _check_fitted(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.A is None or self.B is None or self.c is None:
            raise RuntimeError("Model not fit yet")
        return self.A, self.B, self.c

    def predict_one_step(self, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Teacher-forced one-step predictions: uses true y[t] each step."""
        A, B, c = self._check_fitted()
        y2 = _as_2d(np.asarray(y))
        u2 = _as_2d(np.asarray(u))
        T = y2.shape[0]
        out = np.zeros_like(y2)
        out[0] = y2[0]
        out[1:] = (y2[:-1] @ A.T) + (u2[:-1] @ B.T) + c
        return out

    def predict_open_loop(self, y0: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Open-loop rollout: recursively feeds its own predictions."""
        A, B, c = self._check_fitted()
        u2 = _as_2d(np.asarray(u))
        y0 = np.asarray(y0)
        if y0.ndim == 0:
            y0 = y0.reshape(1)
        if y0.ndim != 1:
            raise ValueError(f"y0 must be 1D, got {y0.shape}")
        ny = A.shape[0]
        if y0.shape[0] != ny:
            raise ValueError(f"y0 has dim {y0.shape[0]} but model ny={ny}")

        T = u2.shape[0]
        out = np.zeros((T, ny), dtype=float)
        out[0] = y0
        for t in range(T - 1):
            out[t + 1] = (A @ out[t]) + (B @ u2[t]) + c
        return out

    @staticmethod
    def nmse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mse = float(np.mean((y_true - y_pred) ** 2))
        var = float(np.var(y_true))
        return mse / (var + eps)

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        return 1.0 - LinearSystemID.nmse(y_true, y_pred)
