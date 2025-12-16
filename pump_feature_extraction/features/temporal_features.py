"""
Compute temporal features for ROI motion segment.

Features include:
- Basic statistics (mean, std, max, sum, median)
- Motion shape (peaks, slopes)
- Temporal dynamics (autocorrelation, CV, entropy, RMS)
- Peak interval mean
- Start/end variance
"""

from typing import Any, Dict, Sequence
import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy, skew, kurtosis
from scipy.signal import find_peaks
from ..utils import autocorr, compute_normalized_slopes, compute_rolling_slopes


def compute_temporal_features(
    segment_motion: NDArray[np.float64],
    fps: int = 30
) -> Dict[str, Any]:

    """
    Compute temporal motion features for a segment of ROI motion.

    Args:
        segment_motion (NDArray[np.float64]): Motion values for a segment of frames.
        fps (int, optional): Frames per second. Defaults to 30.

    Returns:
        Dict[str, Any]: Dictionary of computed temporal features.
    """
    seg = np.asarray(segment_motion, dtype=float)
    out: Dict[str, Any] = {}

    # --- Basic stats ---
    out["motion_mean"] = float(np.mean(seg)) if seg.size else 0.0
    out["motion_std"] = float(np.std(seg)) if seg.size else 0.0
    out["motion_max"] = float(np.max(seg)) if seg.size else 0.0
    out["motion_sum"] = float(np.sum(seg)) if seg.size else 0.0
    out["motion_first"] = float(seg[0]) if seg.size else 0.0
    out["motion_last"] = float(seg[-1]) if seg.size else 0.0

    # --- Distribution shape ---
    if seg.size:
        out["motion_skewness"] = float(skew(seg))
        out["motion_kurtosis"] = float(kurtosis(seg))  # Fisher's definition (0 = normal)
    else:
        out["motion_skewness"] = 0.0
        out["motion_kurtosis"] = 0.0

    # --- RMS ---
    out["motion_rms"] = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0

    # --- Peaks & shape ---
    slope_start, slope_end, norm_start, norm_end = compute_normalized_slopes(seg, motion_rms=out["motion_rms"])
    peaks, _ = find_peaks(seg)
    out["motion_peak_count"] = int(len(peaks))
    out["motion_slope_start"] = slope_start
    out["motion_slope_end"] = slope_end
    out["motion_slope_start_norm"] = norm_start
    out["motion_slope_end_norm"] = norm_end
    out["motion_median"] = float(np.median(seg)) if seg.size else 0.0
    out["motion_iqr"] = float(np.percentile(seg, 75) - np.percentile(seg, 25)) if seg.size else 0.0
    out["motion_mean_abs_diff"] = float(np.mean(np.abs(np.diff(seg)))) if seg.size > 1 else 0.0

    # --- Rolling slope (1-second window) ---
    window = max(3, fps)   # one second, or at least 3 samples
    rolling = compute_rolling_slopes(seg, window)

    if rolling:
        out["motion_roll_slope_mean"] = float(np.mean(rolling))
        out["motion_roll_slope_std"] = float(np.std(rolling))
        out["motion_roll_slope_max"] = float(np.max(rolling))
        out["motion_roll_slope_min"] = float(np.min(rolling))
        out["motion_roll_slope_last"] = float(rolling[-1])
    else:
        out["motion_roll_slope_mean"] = 0.0
        out["motion_roll_slope_std"] = 0.0
        out["motion_roll_slope_max"] = 0.0
        out["motion_roll_slope_min"] = 0.0
        out["motion_roll_slope_last"] = 0.0


    # --- Temporal dynamics ---
    out["motion_autocorr_lag1"] = autocorr(seg, 1)
    out["motion_autocorr_lag2"] = autocorr(seg, 2)
    out["motion_autocorr_lag3"] = autocorr(seg, 3)
    out["motion_cv"] = float(np.std(seg) / (np.mean(seg) + 1e-9)) if seg.size else 0.0

    # --- Entropy ---
    if seg.size and np.sum(seg) > 0:
        probs = seg / (np.sum(seg) + 1e-9)
        out["motion_entropy"] = float(entropy(probs))
    else:
        out["motion_entropy"] = 0.0


    # --- Peak interval mean (seconds) ---
    if len(peaks) > 1:
        out["peak_interval_mean"] = float(np.mean(np.diff(peaks) / fps))
    else:
        out["peak_interval_mean"] = 0.0

    # --- Start / end variance ---
    if seg.size >= 10:
        out["start_var"] = float(np.var(seg[:10]))
        out["end_var"] = float(np.var(seg[-10:]))
    else:
        out["start_var"] = 0.0
        out["end_var"] = 0.0

    return out
