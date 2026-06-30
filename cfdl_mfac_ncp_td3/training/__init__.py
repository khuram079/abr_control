"""Training routines (TD3 with curriculum + domain randomization)."""

from .train_td3 import train_td3, evaluate_policy

__all__ = ["train_td3", "evaluate_policy"]
