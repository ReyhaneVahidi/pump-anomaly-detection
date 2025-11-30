"""
Compute peak geometry features from a motion signal.
"""

from typing import Sequence, Dict
import numpy as np
from scipy.signal import find_peaks


def compute_peak_features(motion_signal: Sequence[float]) -> Dict[str, float]:
    """
    Features:
        - motion_peak_count
        - peak_mean_height
        - peak_std_height
        - peak_max_height
    """
    sig = np.asarray(motion_signal, dtype=float)
    out: Dict[str, float] = {}

    if sig.size == 0:
        return {
            "motion_peak_count": 0,
            "peak_mean_height": 0.0,
            "peak_std_height": 0.0,
            "peak_max_height": 0.0,
        }

    # Peak detection
    peaks, _ = find_peaks(sig)
    out["motion_peak_count"] = int(len(peaks))

    # If no peaks detected, keep simple defaults
    if len(peaks) == 0:
        out["peak_mean_height"] = 0.0
        out["peak_std_height"] = 0.0
        out["peak_max_height"] = 0.0
        return out

    peak_values = sig[peaks]

    out["peak_mean_height"] = float(np.mean(peak_values))
    out["peak_std_height"]  = float(np.std(peak_values))
    out["peak_max_height"]  = float(np.max(peak_values))

    return out
