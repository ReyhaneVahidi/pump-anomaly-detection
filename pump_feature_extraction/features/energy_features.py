"""
Compute energy-based motion features.
"""

from typing import Sequence, Dict
import numpy as np
from numpy.typing import NDArray

def compute_energy_features(motion_signal: NDArray[np.float64]) -> Dict[str, float]:
    """
    Compute energy and power features from a motion signal.

    Features:
        - motion_energy: Sum of squared motion
        - motion_power: Energy normalized by length

    Args:
        motion_signal (NDArray[np.float64]): 1D motion magnitude per frame.

    Returns:
        Dict[str, float]: Dictionary containing energy-based features.
    """
    sig = np.asarray(motion_signal, dtype=float)
    out: Dict[str, float] = {}

    if sig.size == 0:
        return {
            "motion_energy": 0.0,
            "motion_power": 0.0,
            "motion_rms": 0.0,
        }

    # Total energy
    energy = np.sum(sig ** 2)
    out["motion_energy"] = float(energy)

    # Power (normalized)
    out["motion_power"] = float(energy / sig.size)

    return out
