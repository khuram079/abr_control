"""Typed configuration objects for every subsystem of the framework.

Using :mod:`dataclasses` keeps the configuration explicit, self-documenting
and trivially serialisable, while allowing each experiment to override only
the fields it cares about.  All physical defaults correspond to a
REMUS-100-class torpedo AUV (see :mod:`cfdl_mfac_ncp_td3.dynamics`).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Tuple


@dataclass
class SimConfig:
    """Top-level integration / simulation settings."""

    dt: float = 0.05  # integration step [s]
    horizon: float = 60.0  # default episode length [s]
    integrator: str = "rk4"  # 'rk4' or 'euler'
    seed: int = 0

    @property
    def n_steps(self) -> int:
        return int(round(self.horizon / self.dt))


@dataclass
class CurrentConfig:
    """Ocean-current disturbance model."""

    # Mean current expressed in the NED earth frame [m/s].
    mean_velocity: Tuple[float, float, float] = (0.3, 0.1, 0.0)
    # First-order Gauss-Markov turbulence parameters.
    turbulence_intensity: float = 0.05  # std-dev [m/s]
    correlation_time: float = 20.0  # [s]
    enabled: bool = True


@dataclass
class ThrusterConfig:
    """Actuator (thruster + control surface) and allocation settings."""

    max_thrust: float = 50.0  # main propeller thrust limit [N]
    max_fin_force: float = 20.0  # equivalent fin force/moment limit [N or N*m]
    time_constant: float = 0.1  # first-order actuator lag [s]
    rate_limit: float = 200.0  # |d(cmd)/dt| limit [unit/s]
    # Generalised-force saturation, one entry per DOF
    # (surge, sway, heave, roll, pitch, yaw).
    tau_max: Tuple[float, float, float, float, float, float] = (
        50.0,
        30.0,
        30.0,
        10.0,
        20.0,
        20.0,
    )


@dataclass
class MFACConfig:
    """Compact-Form Dynamic Linearization MFAC hyper-parameters.

    Defaults are tuned for the AUV inner *velocity* loop, whose per-step gain
    ``d(nu)/d(tau)`` is small (~1e-3); hence the small ``phi_init`` and ``lam``.
    The Stage-2 unit tests pass their own explicit configs for generic plants.
    """

    eta: float = 1.0  # PJM estimation step size
    mu: float = 1.0  # PJM estimation penalty
    rho: float = 1.0  # control-law step size
    lam: float = 5e-4  # control-law penalty (lambda)
    epsilon: float = 1e-7  # reset threshold
    phi_init: float = 5e-3  # initial diagonal of the pseudo-Jacobian
    u_limit: float = 12.0  # per-channel |Δtau| clip [N or N*m]


@dataclass
class ObserverConfig:
    """Observer-suite settings."""

    # State observer (high-gain / Kalman blend).
    process_noise: float = 1e-3
    meas_noise: float = 1e-2
    # Nonlinear disturbance observer gain.
    ndob_gain: float = 5.0
    # Fault observer (effectiveness estimation).
    fault_adapt_gain: float = 2.0
    fault_forgetting: float = 0.999


@dataclass
class TD3Config:
    """Twin-Delayed DDPG hyper-parameters."""

    hidden_sizes: Tuple[int, int] = (256, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    exploration_noise: float = 0.1
    batch_size: int = 256
    buffer_size: int = 1_000_000
    warmup_steps: int = 1000
    device: str = "cpu"


@dataclass
class NCPConfig:
    """Neural Circuit Policy (liquid network) supervisor settings."""

    n_sensory: int = 8
    n_inter: int = 12
    n_command: int = 6
    n_motor: int = 1  # outputs a blending coefficient in [0, 1]
    sensory_fanout: int = 4
    inter_fanout: int = 4
    recurrent_command: int = 3
    motor_fanin: int = 4
    ode_unfolds: int = 6  # semi-implicit Euler sub-steps for the LTC cell
    seed: int = 1


@dataclass
class SupervisorConfig:
    """Confidence-guided fusion supervisor."""

    # Weights mapping normalised error / disturbance / fault signals to a
    # scalar confidence in the model-free adaptive controller.
    w_error: float = 4.0
    w_disturbance: float = 2.0
    w_fault: float = 3.0
    blend_floor: float = 0.0  # minimum RL authority
    blend_ceiling: float = 1.0  # maximum RL authority
    use_ncp: bool = True  # gate the blend through the NCP supervisor
    smoothing: float = 0.1  # low-pass on the blending coefficient

    # --- competence-aware trust gate ---------------------------------- #
    # RL authority is granted by MFAC *distress* only; without a check on
    # whether the learned policy is actually helping, a bad policy would be
    # trusted (and can degrade tracking below pure MFAC).  The trust gate
    # scales authority by an online estimate of the RL policy's competence:
    # trust rises while RL holds authority and the error keeps falling, and
    # falls when the error grows under RL authority.  Final authority is
    # ``alpha = alpha_distress * trust``, so an unhelpful policy loses
    # authority and the hybrid gracefully falls back to MFAC.
    competence_gating: bool = True
    trust_init: float = 0.5  # initial (skeptical) RL trust
    trust_rate: float = 0.03  # trust adaptation rate per step
    trust_ema_beta: float = 0.05  # error EMA smoothing for the trend test
    trust_active_thresh: float = 0.15  # alpha above which RL is "responsible"


@dataclass
class ExperimentConfig:
    """Aggregate configuration passed around the framework."""

    sim: SimConfig = field(default_factory=SimConfig)
    current: CurrentConfig = field(default_factory=CurrentConfig)
    thruster: ThrusterConfig = field(default_factory=ThrusterConfig)
    mfac: MFACConfig = field(default_factory=MFACConfig)
    observer: ObserverConfig = field(default_factory=ObserverConfig)
    td3: TD3Config = field(default_factory=TD3Config)
    ncp: NCPConfig = field(default_factory=NCPConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def default_config() -> ExperimentConfig:
    """Return a fresh default :class:`ExperimentConfig`."""

    return ExperimentConfig()
