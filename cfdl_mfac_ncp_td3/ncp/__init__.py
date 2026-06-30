"""Neural Circuit Policy supervisor: sparse wiring + Liquid Time-Constant cell."""

from .wiring import NCPWiring
from .liquid_network import LiquidNetwork

__all__ = ["NCPWiring", "LiquidNetwork"]
