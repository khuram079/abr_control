"""Reference trajectory generators for AUV tracking tasks.

Every trajectory exposes ``reference(t) -> (eta_d, eta_d_dot)`` with the
6-DOF desired pose and its time derivative in the NED frame, plus ``duration``.
"""

from .trajectories import (
    Trajectory,
    SetpointTrajectory,
    SinusoidalTrajectory,
    HelixTrajectory,
    WaypointTrajectory,
    LawnmowerTrajectory,
    make_trajectory,
    TRAJECTORY_REGISTRY,
)

__all__ = [
    "Trajectory",
    "SetpointTrajectory",
    "SinusoidalTrajectory",
    "HelixTrajectory",
    "WaypointTrajectory",
    "LawnmowerTrajectory",
    "make_trajectory",
    "TRAJECTORY_REGISTRY",
]
