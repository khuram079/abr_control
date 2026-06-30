"""Model-free adaptive control: CFDL pseudo-Jacobian estimation + MFAC law."""

from .pseudo_gradient import PseudoJacobianEstimator
from .cfdl import CFDLModel
from .mfac import CFDLMFAC

__all__ = ["PseudoJacobianEstimator", "CFDLModel", "CFDLMFAC"]
