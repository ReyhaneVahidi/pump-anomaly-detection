"""
Compute brightness-based features from a segment of frames.
"""

from typing import Sequence, Dict
import numpy as np
from ..utils import safe_slope


def compute_brightness_features(segment_brightness: Sequence[float]) -> Dict[str, float]:
    """
    Compute basic brightness features.

    Features:
        - brightness_mean: Average brightness
        - brightness_std: Standard deviation of brightness
        - brightness_slope: Trend (slope) over time

    Args:
        segment_brightness (Sequence[float]): Mean brightness per frame for a segment.

    Returns:
        Dict[str, float]: Dictionary with brightness features.
    """
    seg = np.asarray(segment_brightness, dtype=float)
    out: Dict[str, float] = {}
    out["brightness_mean"] = float(np.mean(seg)) if seg.size else 0.0
    out["brightness_std"] = float(np.std(seg)) if seg.size else 0.0
    out["brightness_slope"] = safe_slope(seg)
    return out
