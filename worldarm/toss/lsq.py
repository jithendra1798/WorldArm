"""Small nonlinear least squares (Levenberg-Marquardt with a numeric Jacobian; no scipy in the venv), with
the parameters' standard deviations, for the profile fit and the throw calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Fit:
    x: np.ndarray            # best parameters
    sd: np.ndarray           # their standard deviations (residual-scaled; inf where not determined)
    rms: float               # residual root mean square
    residuals: np.ndarray


def least_squares(f: Callable[[np.ndarray], np.ndarray], x0, iters: int = 100, rel_step: float = 1e-6) -> Fit:
    """Minimise sum(f(x)^2) from x0."""
    x = np.asarray(x0, float).copy()
    r = f(x)
    lam = 1e-3

    def jac(x, r):
        J = np.empty((len(r), len(x)))
        for i in range(len(x)):
            h = rel_step * max(1.0, abs(x[i]))
            xh = x.copy()
            xh[i] += h
            J[:, i] = (f(xh) - r) / h
        return J

    J = jac(x, r)
    for _ in range(iters):
        A, g = J.T @ J, J.T @ r
        step = np.linalg.solve(A + lam * np.diag(np.maximum(np.diag(A), 1e-12)), -g)
        x_new = x + step
        r_new = f(x_new)
        if np.all(np.isfinite(r_new)) and r_new @ r_new < r @ r:
            done = np.linalg.norm(step) <= 1e-10 * (np.linalg.norm(x) + 1e-10)
            x, r, lam = x_new, r_new, max(lam / 3.0, 1e-12)
            J = jac(x, r)
            if done:
                break
        else:
            lam *= 10.0
            if lam > 1e12:
                break
    dof = max(1, len(r) - len(x))
    s2 = float(r @ r) / dof
    try:
        cov = np.linalg.inv(J.T @ J) * s2
        sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        sd = np.full(len(x), np.inf)
    return Fit(x, sd, float(np.sqrt(np.mean(r * r))) if len(r) else 0.0, r)
