"""
Compute peak geometry + rhythm features from a motion signal.
"""

from typing import Dict
import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks


def compute_peak_features(motion_signal: NDArray[np.float64], fps: int = 30) -> Dict[str, float]:
    sig = np.asarray(motion_signal, dtype=float)

    out = {
        "motion_peak_count": 0,
        "peak_mean_height": 0.0,
        "peak_std_height": 0.0,
        "peak_max_height": 0.0,
        "motion_peak_prom_mean": 0.0,
        "motion_peak_prom_max": 0.0,
        "peak_interval_mean": 0.0,
        "peak_interval_std": 0.0,
        "peak_freq_from_interval": 0.0,
    }

    if sig.size < 3:
        return out

    # --- Adaptive peak settings (scale-robust) ---
    # Use variability-based prominence so it works across different normalization/scales.
    sig_std = float(np.std(sig))
    if sig_std < 1e-12:
        return out

    prominence = 0.5 * sig_std        # tune factor later (0.3–1.0 typical)
    distance = max(1, int(0.1 * fps)) # at least 0.1s between peaks

    peaks, props = find_peaks(sig, distance=distance, prominence=prominence)

    out["motion_peak_count"] = int(len(peaks))
    if len(peaks) == 0:
        return out

    peak_values = sig[peaks]
    out["peak_mean_height"] = float(np.mean(peak_values))
    out["peak_std_height"] = float(np.std(peak_values, ddof=1)) if len(peak_values) > 1 else 0.0
    out["peak_max_height"] = float(np.max(peak_values))

    prom = props.get("prominences", None)
    if prom is not None and len(prom) > 0:
        out["motion_peak_prom_mean"] = float(np.mean(prom))
        out["motion_peak_prom_max"] = float(np.max(prom))

    # --- Peak timing (rhythm) ---
    if len(peaks) > 1:
        intervals = np.diff(peaks) / float(fps)
        mean_period = float(np.mean(intervals))
        out["peak_interval_mean"] = mean_period
        out["peak_interval_std"] = float(np.std(intervals, ddof=1)) if len(intervals) > 1 else 0.0
        out["peak_freq_from_interval"] = 1.0 / (mean_period + 1e-9)

    return out
