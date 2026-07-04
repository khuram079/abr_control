"""Model-free adaptive control: CFDL pseudo-Jacobian estimation + MFAC law."""

from .pseudo_gradient import PseudoJacobianEstimator
from .cfdl import CFDLModel
from .mfac import CFDLMFAC
from .pfdl import PFDLMFAC
from .confidence import ConfidenceEstimator
from .supervisor import HybridSupervisor
from .hybrid_controller import HybridController

__all__ = [
    "PseudoJacobianEstimator",
    "CFDLModel",
    "CFDLMFAC",
    "PFDLMFAC",
    "ConfidenceEstimator",
    "HybridSupervisor",
    "HybridController",
]
