"""The five formation-control performance indicators.

Computed from a :meth:`FormationSimulator.simulate` result:

1. ``formation_rmse``  -- RMS follower position error to the assigned slot [m]
2. ``attitude_rmse``   -- RMS follower attitude error [rad]
3. ``recovery_time``   -- mean time a follower spends recovering formation [s]
4. ``control_energy``  -- total integrated squared wrench across followers
5. ``max_deviation``   -- peak follower formation error [m] (overshoot / stability)
"""

from __future__ import annotations

import numpy as np

INDICATORS = ("formation_rmse", "attitude_rmse", "recovery_time",
              "control_energy", "max_deviation")


def indicators(result: dict) -> dict:
    fe = np.asarray(result["form_err"])          # (T, n)
    ae = np.asarray(result["att_err"])           # (T, n)
    return {
        "formation_rmse": float(np.sqrt(np.mean(fe ** 2))),
        "attitude_rmse": float(np.sqrt(np.mean(ae ** 2))),
        "recovery_time": float(np.mean(result["recovery_time"])),
        "control_energy": float(np.sum(result["energy"])),
        "max_deviation": float(np.max(fe)),
    }
