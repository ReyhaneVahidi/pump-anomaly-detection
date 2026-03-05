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
    out["motion_max"] = float(np.max(seg)) if seg.size else 0.0 # max is sensitive to outliers, but can be informative for pump-on peaks.
    # Robust Max (95th percentile ignores one-frame glitches)
    out["motion_max_robust"] = float(np.percentile(seg, 95)) if seg.size else 0.0 
    # out["motion_sum"] = float(np.sum(seg)) if seg.size else 0.0 #redundant with mean and duration, motion_mean × number_of_frames
    out["motion_start_avg"] = float(np.mean(seg[:10]) if seg.size >= 10 else 0.0)
    out["motion_end_avg"] = float(np.mean(seg[-10:]) if seg.size >= 10 else 0.0) 

    # --- Distribution shape --- not that useful. maybe remove later.
    # if seg.size >= 3 and np.std(seg) > 1e-8:
    #     out["motion_skewness"] = float(skew(seg, bias=True)) # measures how symmetrical the motion signal is.
    #     out["motion_kurtosis"] = float(kurtosis(seg, fisher=True, bias=True)) # tells about the "extremes" in data.
    # else:
    #     out["motion_skewness"] = 0.0
    #     out["motion_kurtosis"] = 0.0


    # --- RMS ---
    # "quadratric mean." It is very similar to the motion_mean, but it gives more "weight" to higher values.
    out["motion_rms"] = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
    
    out["motion_crest_factor"] = out["motion_max"] / (out["motion_rms"] + 1e-9) if seg.size else 0.0
    out["motion_crest_factor_p95"] = out["motion_max_robust"] / (out["motion_rms"] + 1e-9) if seg.size else 0.0



    # --- shape ---
    slope_start, slope_end, norm_start, norm_end = compute_normalized_slopes(
        seg, motion_rms=out["motion_rms"], fps=fps
    )

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
    # out["motion_autocorr_lag1"] = autocorr(seg, 1)
    # out["motion_autocorr_lag2"] = autocorr(seg, 2)
    out["motion_autocorr_lag3"] = autocorr(seg, 3)
    out["motion_cv"] = float(np.std(seg) / (np.mean(seg) + 1e-9)) if seg.size else 0.0

    # --- Entropy --- (not that useful, maybe remove later, mostly the same number)
    # if seg.size and np.sum(seg) > 0:
    #     probs = seg / (np.sum(seg) + 1e-9)
    #     out["motion_entropy"] = float(entropy(probs))
    # else:
    #     out["motion_entropy"] = 0.0

    # --- Start / end variance ---
    if seg.size >= 10:
        out["start_var"] = float(np.var(seg[:10]))
        out["end_var"] = float(np.var(seg[-10:]))
    else:
        out["start_var"] = 0.0
        out["end_var"] = 0.0

    return out
