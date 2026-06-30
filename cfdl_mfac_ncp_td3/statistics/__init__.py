"""Monte-Carlo evaluation and statistical-significance analysis."""

from .montecarlo import monte_carlo
from .tests import (
    cohens_d,
    confidence_interval,
    paired_ttest,
    wilcoxon,
    compare,
)

__all__ = [
    "monte_carlo",
    "cohens_d",
    "confidence_interval",
    "paired_ttest",
    "wilcoxon",
    "compare",
]
