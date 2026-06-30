"""Publication-quality plotting utilities (headless / Agg backend)."""

from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt  # noqa: E402

DOF_LABELS = ["x [m]", "y [m]", "z [m]", "roll [rad]", "pitch [rad]", "yaw [rad]"]

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.6,
})


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)


def plot_trajectory_3d(log: dict, path: str, title: str = "AUV trajectory") -> str:
    eta = np.asarray(log["eta"])
    eta_d = np.asarray(log["eta_d"])
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(eta_d[:, 0], eta_d[:, 1], -eta_d[:, 2], "k--", label="reference")
    ax.plot(eta[:, 0], eta[:, 1], -eta[:, 2], "C0-", label="actual")
    ax.scatter(*[[eta[0, i]] for i in range(2)], [-eta[0, 2]], c="g", s=40, label="start")
    ax.set_xlabel("North [m]"); ax.set_ylabel("East [m]"); ax.set_zlabel("Up [m]")
    ax.set_title(title); ax.legend()
    _ensure_dir(path); fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_tracking_errors(log: dict, path: str, title: str = "Tracking error") -> str:
    t = np.asarray(log["t"]); err = np.asarray(log["error"])
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for i, ax in enumerate(axes.ravel()):
        ax.plot(t, err[:, i], "C3")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel(DOF_LABELS[i]); ax.set_xlabel("time [s]")
    fig.suptitle(title)
    _ensure_dir(path); fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_control_effort(log: dict, path: str, title: str = "Control effort") -> str:
    t = np.asarray(log["t"]); tau = np.asarray(log["tau"])
    labels = ["X", "Y", "Z", "K", "M", "N"]
    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(6):
        ax.plot(t, tau[:, i], label=labels[i])
    ax.set_xlabel("time [s]"); ax.set_ylabel("generalised force / moment")
    ax.set_title(title); ax.legend(ncol=6, fontsize=8)
    _ensure_dir(path); fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_blending(log: dict, path: str, title: str = "RL authority (alpha)") -> str:
    t = np.asarray(log["t"]); alpha = np.asarray(log["alpha"])
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, alpha, "C2")
    ax.fill_between(t, 0, alpha, alpha=0.2, color="C2")
    ax.set_ylim(-0.05, 1.05); ax.set_xlabel("time [s]"); ax.set_ylabel("alpha")
    ax.set_title(title)
    _ensure_dir(path); fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path


def plot_comparison_bars(summaries: dict, metric: str, path: str,
                         title: str | None = None) -> str:
    """Bar chart with error bars comparing a metric across controllers.

    ``summaries`` maps controller name -> {metric: {"mean":..,"std":..}}.
    """

    names = list(summaries.keys())
    means = [summaries[n][metric]["mean"] for n in names]
    stds = [summaries[n][metric]["std"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=4, color="C0", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel(metric); ax.set_title(title or f"{metric} comparison")
    _ensure_dir(path); fig.tight_layout(); fig.savefig(path); plt.close(fig)
    return path
