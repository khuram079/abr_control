"""Tracking-performance metrics for AUV control evaluation."""

from __future__ import annotations

import numpy as np

POS_IDX = slice(0, 3)
ATT_IDX = slice(3, 6)


def rmse(errors: np.ndarray, idx=None) -> float:
    """Root-mean-square error norm over time (optionally a DOF subset)."""

    e = np.asarray(errors, dtype=float)
    e = e if idx is None else e[:, idx]
    return float(np.sqrt(np.mean(np.sum(e ** 2, axis=1))))


def iae(errors: np.ndarray, dt: float) -> float:
    """Integral of absolute error (per-step error norm integrated)."""

    e = np.asarray(errors, dtype=float)
    return float(np.sum(np.linalg.norm(e, axis=1)) * dt)


def ise(errors: np.ndarray, dt: float) -> float:
    """Integral of squared error."""

    e = np.asarray(errors, dtype=float)
    return float(np.sum(np.sum(e ** 2, axis=1)) * dt)


def max_error(errors: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(np.asarray(errors, dtype=float), axis=1)))


def control_energy(taus: np.ndarray, dt: float) -> float:
    """Integrated squared control effort (a proxy for energy consumption)."""

    u = np.asarray(taus, dtype=float)
    return float(np.sum(np.sum(u ** 2, axis=1)) * dt)


def control_smoothness(taus: np.ndarray) -> float:
    """Total variation of the command (lower is smoother)."""

    u = np.asarray(taus, dtype=float)
    if len(u) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(u, axis=0), axis=1)))


def settling_time(errors: np.ndarray, dt: float, threshold: float = 0.1) -> float:
    """Time after which the error norm stays below ``threshold`` (NaN if never)."""

    norms = np.linalg.norm(np.asarray(errors, dtype=float), axis=1)
    below = norms < threshold
    for k in range(len(below)):
        if np.all(below[k:]):
            return float(k * dt)
    return float("nan")


def summarize(errors: np.ndarray, taus: np.ndarray, dt: float) -> dict:
    """Return the full metric dictionary for one rollout."""

    return {
        "rmse": rmse(errors),
        "rmse_pos": rmse(errors, POS_IDX),
        "rmse_att": rmse(errors, ATT_IDX),
        "iae": iae(errors, dt),
        "ise": ise(errors, dt),
        "max_error": max_error(errors),
        "control_energy": control_energy(taus, dt),
        "smoothness": control_smoothness(taus),
        "settling_time": settling_time(errors, dt),
    }
