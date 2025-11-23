"""
Utility functions for motion and feature analysis.
"""

from typing import Sequence, List, Tuple
import numpy as np
from scipy.stats import linregress



def autocorr(x: Sequence[float], lag: int) -> float:
    """
    Compute the autocorrelation of a 1D sequence at a given lag.

    Args:
        x (Sequence[float]): Input sequence (e.g., motion values).
        lag (int): Lag value to compute autocorrelation.

    Returns:
        float: Correlation coefficient at the given lag.
               Returns 0.0 if the sequence is too short.
    """
    x_array = np.asarray(x, dtype=float)
    if len(x_array) <= lag:
        return 0.0
    return float(np.corrcoef(x_array[:-lag], x_array[lag:])[0, 1])

def safe_slope(values: Sequence[float], min_len: int = 5) -> float:
    """
    Compute the slope of a sequence using linear regression.
    Returns 0.0 if the sequence is too short.

    Args:
        values (Sequence[float]): Input sequence.
        min_len (int, optional): Minimum length required to compute slope. Defaults to 5.

    Returns:
        float: Slope of the sequence.
    """
    values_array = np.asarray(values, dtype=float)
    if values_array.size < min_len:
        return 0.0
    t = np.arange(len(values_array))
    return float(linregress(t, values_array).slope)


def compute_slope(signal: np.ndarray) -> float:
    """Return the slope of a 1D signal using linear regression."""
    x = np.arange(len(signal))
    slope, _, _, _, _ = linregress(x, signal)
    return float(slope)

def compute_normalized_slopes(
    signal: np.ndarray, motion_rms: float, pct: float = 0.2
) -> Tuple[float, float, float, float]:
    """
    Compute slope_start and slope_end using pct of the signal,
    and normalize each by motion_rms.

    Returns:
        slope_start, slope_end, norm_start, norm_end
    """
    n = len(signal)
    k = max(3, int(n * pct))

    start_segment = signal[:k]
    end_segment   = signal[-k:]

    slope_start = compute_slope(start_segment)
    slope_end   = compute_slope(end_segment)

    if motion_rms == 0:
        norm_start, norm_end = 0.0, 0.0
    else:
        norm_start = slope_start / motion_rms
        norm_end   = slope_end / motion_rms

    return slope_start, slope_end, norm_start, norm_end

def compute_rolling_slopes(signal: Sequence[float], window: int) -> List[float]:
    """
    Compute rolling slopes over a sliding window.

    Args:
        signal (Sequence[float]): Input 1D signal.
        window (int): Number of samples per window.

    Returns:
        List[float]: Slope value for each window step.
    """
    sig = np.asarray(signal, dtype=float)
    if window < 3 or sig.size < window:
        return []

    slopes = []
    for i in range(len(sig) - window + 1):
        segment = sig[i:i + window]
        x = np.arange(window)
        slope, _, _, _, _ = linregress(x, segment)
        slopes.append(slope)

    return slopes