"""Leader--follower formation control, recovery, metrics and fair tuning."""

from .formation_env import FormationSimulator, FormationConfig
from .metrics import indicators, INDICATORS
from .tuning import tune_all, tune_controller, build_tuned, SEARCH_SPACES

__all__ = [
    "FormationSimulator",
    "FormationConfig",
    "indicators",
    "INDICATORS",
    "tune_all",
    "tune_controller",
    "build_tuned",
    "SEARCH_SPACES",
]
