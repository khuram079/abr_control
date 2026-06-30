"""CFDL-MFAC-NCP-TD3: a hybrid intelligent control framework for AUVs.

This package implements a research-grade, modular framework that combines:

* a 6-DOF REMUS-class autonomous underwater vehicle (AUV) simulator,
* a model-free adaptive controller based on Compact-Form Dynamic
  Linearization (CFDL-MFAC),
* a suite of observers (state / disturbance / fault),
* a Neural Circuit Policy (NCP / liquid network) supervisor,
* a Twin-Delayed DDPG (TD3) reinforcement-learning agent,
* a confidence-guided fusion supervisor that blends the model-free
  adaptive controller and the learned policy, and
* a set of classical benchmark controllers (PID, SMC, backstepping, MPC).

The framework is intentionally built in verified stages; every module is
importable and unit-tested in isolation so the system can be developed,
validated and extended incrementally.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
