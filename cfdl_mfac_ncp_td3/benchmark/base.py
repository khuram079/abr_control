"""Common interface and helpers for the benchmark controllers.

All benchmark controllers act through the same fully-actuated 6-DOF
generalised-force interface as the hybrid controller, so comparisons are fair:

    tau = controller.control(eta, nu, eta_d, eta_d_dot, dt)

with ``eta = [x,y,z,phi,theta,psi]`` (NED pose), ``nu`` the body-frame
velocity, and ``eta_d`` / ``eta_d_dot`` the desired pose and its rate.
"""

from __future__ import annotations

import numpy as np

ANGLE_IDX = (3, 4, 5)


def pose_error(eta_d: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """Pose error ``eta_d - eta`` with angle channels wrapped to (-pi, pi]."""

    e = np.asarray(eta_d, dtype=float).reshape(6) - np.asarray(eta, dtype=float).reshape(6)
    e[list(ANGLE_IDX)] = (e[list(ANGLE_IDX)] + np.pi) % (2 * np.pi) - np.pi
    return e


class BaseController:
    """Abstract benchmark controller."""

    name = "base"

    def reset(self) -> None:  # pragma: no cover - trivial
        pass

    def control(self, eta, nu, eta_d, eta_d_dot=None, dt: float = 0.05) -> np.ndarray:
        raise NotImplementedError
