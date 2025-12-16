"""
Compute frequency-domain features from a segment of motion values.
"""

from typing import Sequence, Dict, Any
import numpy as np
from numpy.typing import NDArray


def compute_frequency_features(
    segment_motion: NDArray[np.float64],
    fps: int = 30
) -> Dict[str, float]:
    """
    Compute frequency-domain features from a motion segment using FFT.

    Features:
        - dominant_freq: Frequency with maximum magnitude
        - spectral_centroid: Weighted mean frequency
        - spectral_bandwidth: Standard deviation of frequencies

    Args:
        segment_motion (NDArray[np.float64]): Motion values for a segment of frames.
        fps (int, optional): Frames per second. Defaults to 30.

    Returns:
        Dict[str, float]: Dictionary with frequency-domain features.
    """
    seg = np.asarray(segment_motion, dtype=float)
    out: Dict[str, float] = {}

    if seg.size < 2:
        out["dominant_freq"] = 0.0
        out["spectral_centroid"] = 0.0
        out["spectral_bandwidth"] = 0.0
        return out

    fft_vals = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(len(seg), d=1.0 / fps)

    if np.sum(fft_vals) <= 0:
        out["dominant_freq"] = 0.0
        out["spectral_centroid"] = 0.0
        out["spectral_bandwidth"] = 0.0
        return out

    spectral_centroid = float(np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-9))
    spectral_bandwidth = float(
        np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft_vals) / (np.sum(fft_vals) + 1e-9))
    )
    dominant_freq = float(freqs[np.argmax(fft_vals)])

    out["dominant_freq"] = dominant_freq
    out["spectral_centroid"] = spectral_centroid
    out["spectral_bandwidth"] = spectral_bandwidth
    return out
