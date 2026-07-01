"""Statistical-significance utilities for controller comparison.

Provides paired/independent hypothesis tests, non-parametric alternatives,
effect sizes and confidence intervals -- the analysis expected when reporting
controller comparisons in a publication.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy import stats as _stats

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size (pooled standard deviation)."""

    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
                     / max(na + nb - 2, 1))
    if pooled < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def confidence_interval(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided ``(1-alpha)`` CI for the mean (t-based, falls back to normal)."""

    x = np.asarray(x, float)
    n = len(x)
    m, se = float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(n))
    if _HAS_SCIPY:
        tcrit = _stats.t.ppf(1 - alpha / 2, n - 1)
    else:  # pragma: no cover
        tcrit = 1.96
    return (m - tcrit * se, m + tcrit * se)


def paired_ttest(a: np.ndarray, b: np.ndarray) -> dict:
    """Paired t-test between two metric vectors (same trials)."""

    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    # Degenerate case: identical samples (e.g. an ablation variant that
    # reduces to the reference) -> no difference, well-defined result.
    if np.allclose(diff, 0.0):
        return {"t": 0.0, "p": 1.0, "cohens_d": 0.0, "mean_diff": 0.0}
    if _HAS_SCIPY:
        t, p = _stats.ttest_rel(a, b)
    else:  # pragma: no cover
        t = np.mean(diff) / (np.std(diff, ddof=1) / np.sqrt(len(diff)))
        p = float("nan")
    return {"t": float(t), "p": float(p), "cohens_d": cohens_d(a, b),
            "mean_diff": float(np.mean(diff))}


def wilcoxon(a: np.ndarray, b: np.ndarray) -> dict:
    """Wilcoxon signed-rank test (non-parametric paired alternative)."""

    a, b = np.asarray(a, float), np.asarray(b, float)
    if not _HAS_SCIPY:  # pragma: no cover
        return {"stat": float("nan"), "p": float("nan")}
    if np.allclose(a - b, 0.0):  # identical samples -> no difference
        return {"stat": 0.0, "p": 1.0}
    try:
        stat, p = _stats.wilcoxon(a, b)
    except ValueError:  # e.g. zero-only differences after filtering
        return {"stat": 0.0, "p": 1.0}
    return {"stat": float(stat), "p": float(p)}


def compare(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray,
            alpha: float = 0.05) -> dict:
    """Full pairwise comparison report for one metric across trials."""

    a, b = np.asarray(a, float), np.asarray(b, float)
    tt = paired_ttest(a, b)
    return {
        "a": name_a,
        "b": name_b,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "ci_a": confidence_interval(a, alpha),
        "ci_b": confidence_interval(b, alpha),
        "ttest": tt,
        "wilcoxon": wilcoxon(a, b),
        "significant": bool(tt["p"] < alpha) if np.isfinite(tt["p"]) else False,
    }
