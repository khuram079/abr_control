"""Concrete reference trajectories for 6-DOF AUV tracking."""

from __future__ import annotations

import numpy as np


class Trajectory:
    """Base class: ``reference(t)`` returns desired pose and pose rate."""

    duration: float = 60.0
    name: str = "base"

    def reference(self, t: float):
        raise NotImplementedError


class SetpointTrajectory(Trajectory):
    """Constant pose setpoint (step response / regulation)."""

    name = "setpoint"

    def __init__(self, target=(3.0, 2.0, 1.5, 0.0, 0.0, 0.5), duration: float = 60.0):
        self.target = np.asarray(target, dtype=float).reshape(6)
        self.duration = duration

    def reference(self, t: float):
        return self.target.copy(), np.zeros(6)


class SinusoidalTrajectory(Trajectory):
    """Independent per-DOF sinusoids (rich excitation)."""

    name = "sinusoidal"

    def __init__(self, amp=(2, 2, 1, 0, 0.2, 0.5), freq=(0.05, 0.04, 0.03, 0, 0.02, 0.03),
                 duration: float = 60.0):
        self.amp = np.asarray(amp, dtype=float).reshape(6)
        self.w = 2 * np.pi * np.asarray(freq, dtype=float).reshape(6)
        self.duration = duration

    def reference(self, t: float):
        eta_d = self.amp * np.sin(self.w * t)
        eta_d_dot = self.amp * self.w * np.cos(self.w * t)
        return eta_d, eta_d_dot


class HelixTrajectory(Trajectory):
    """Descending helix: a classic AUV survey manoeuvre."""

    name = "helix"

    def __init__(self, radius=3.0, pitch=0.1, omega=0.1, duration: float = 60.0):
        self.r, self.pitch, self.w = radius, pitch, omega
        self.duration = duration

    def reference(self, t: float):
        x = self.r * np.cos(self.w * t)
        y = self.r * np.sin(self.w * t)
        z = self.pitch * t
        psi = np.arctan2(np.cos(self.w * t), -np.sin(self.w * t))  # tangent heading
        eta_d = np.array([x, y, z, 0.0, 0.0, psi])
        dx = -self.r * self.w * np.sin(self.w * t)
        dy = self.r * self.w * np.cos(self.w * t)
        eta_d_dot = np.array([dx, dy, self.pitch, 0.0, 0.0, 0.0])
        return eta_d, eta_d_dot


class WaypointTrajectory(Trajectory):
    """Piecewise-linear interpolation through a list of pose waypoints."""

    name = "waypoint"

    def __init__(self, waypoints=None, seg_time: float = 12.0):
        if waypoints is None:
            waypoints = [
                [0, 0, 0, 0, 0, 0],
                [5, 0, 2, 0, 0, 0],
                [5, 5, 2, 0, 0, np.pi / 2],
                [0, 5, 4, 0, 0, np.pi],
                [0, 0, 4, 0, 0, 0],
            ]
        self.wp = np.asarray(waypoints, dtype=float)
        self.seg_time = seg_time
        self.duration = seg_time * (len(self.wp) - 1)

    def reference(self, t: float):
        seg = min(int(t // self.seg_time), len(self.wp) - 2)
        frac = (t - seg * self.seg_time) / self.seg_time
        frac = float(np.clip(frac, 0.0, 1.0))
        a, b = self.wp[seg], self.wp[seg + 1]
        eta_d = a + frac * (b - a)
        eta_d_dot = (b - a) / self.seg_time
        return eta_d, eta_d_dot


class LawnmowerTrajectory(Trajectory):
    """Boustrophedon ("lawnmower") seabed-survey pattern at fixed depth."""

    name = "lawnmower"

    def __init__(self, length=10.0, spacing=2.0, n_lanes=4, speed=0.8,
                 depth=3.0, duration: float = 80.0):
        self.length, self.spacing, self.n_lanes = length, spacing, n_lanes
        self.speed, self.depth = speed, depth
        self.lane_time = length / speed
        self.turn_time = spacing / speed
        self.duration = duration

    def reference(self, t: float):
        period = self.lane_time + self.turn_time
        lane = int(t // period) % self.n_lanes
        local = t - (t // period) * period
        y = lane * self.spacing
        if local < self.lane_time:
            frac = local / self.lane_time
            x = frac * self.length if lane % 2 == 0 else (1 - frac) * self.length
            vx = self.speed if lane % 2 == 0 else -self.speed
            vy = 0.0
        else:
            x = self.length if lane % 2 == 0 else 0.0
            frac = (local - self.lane_time) / self.turn_time
            y = lane * self.spacing + frac * self.spacing
            vx, vy = 0.0, self.spacing / self.turn_time
        psi = np.arctan2(vy, vx) if (abs(vx) + abs(vy)) > 1e-6 else 0.0
        eta_d = np.array([x, y, self.depth, 0.0, 0.0, psi])
        eta_d_dot = np.array([vx, vy, 0.0, 0.0, 0.0, 0.0])
        return eta_d, eta_d_dot


TRAJECTORY_REGISTRY = {
    "setpoint": SetpointTrajectory,
    "sinusoidal": SinusoidalTrajectory,
    "helix": HelixTrajectory,
    "waypoint": WaypointTrajectory,
    "lawnmower": LawnmowerTrajectory,
}


def make_trajectory(name: str, **kwargs) -> Trajectory:
    if name not in TRAJECTORY_REGISTRY:
        raise KeyError(f"unknown trajectory '{name}', choices: {list(TRAJECTORY_REGISTRY)}")
    return TRAJECTORY_REGISTRY[name](**kwargs)
