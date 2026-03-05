"""
flow_window_features.py

Module for windowed feature extraction from ROI-based optical flow signals. 
Designed for integration into automated processing pipelines to quantify 
pump-on segment characteristics.

Input: 
    - 1D optical flow signal (positive, per-frame) for a pump-on segment.
Output: 
    - Dictionary of aggregated per-video features (Medians, IQR, Maxima).

Core Features Extracted:
    - Duty Cycle: Identifies intermittent stops or OFF states within windows.
    - Longest Pause: Calculates the maximum sustained 'off' duration (seconds).
    - Dominant Frequency (Hz): Band-limited FFT analysis of pump rhythm.
    - Spectral Power Ratio: Quantifies peak strength within the target band.
    - Jitter (CV): Signal stability via Coefficient of Variation (Std/Mean).
    - Intensity Metrics: Window-level statistics (Median/IQR of means).
    - Sanity Check: Diff-based peak frequency verification.

Note: This module is purely functional and contains no plotting or I/O operations.

Integration Example:
    from pump_feature_extraction.features.flow_window_features import compute_flow_window_features
    
    # After computing segment_flow in main execution
    flow_stats = compute_flow_window_features(segment_flow, fps=FPS)
    features.update(flow_stats)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks


# ============================================================
# CONFIG (tune here or later move to config.py)
# ============================================================

WINDOW_SIZE_SEC = 2.0
STEP_SIZE_SEC = 1.0

# OFF threshold on RAW flow (positive signal)
OFF_THRESH_METHOD = "median_frac"   # "median_frac" or "percentile"
OFF_THRESH_FRAC_OF_MEDIAN = 0.20
OFF_THRESH_PERCENTILE = 10

# Consider a window "on" if duty_cycle >= this
ON_WINDOW_DUTY_THRESH = 0.80

# FFT band for pump rhythm (set from your observed normal range)
FREQ_MIN_HZ = 5.0
FREQ_MAX_HZ = 15.0
USE_HANN_WINDOW = True

# Peak sanity check (only if you pass diff_signal)
ENABLE_PEAK_SANITYCHECK_DEFAULT = True
PEAK_MIN_PROM_FRAC = 0.5
PEAK_MIN_DIST_HZ = 15  # max expected frequency for distance = fps / this


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _robust_scale_signal(x: np.ndarray, method: str = "mad") -> np.ndarray:
    """
    Robust scaling is optional. We keep it here in case you later want
    "scaled" window features too. Currently we do NOT use scaled for duty/FFT.
    """
    x = x.astype(np.float32)
    if method == "none":
        return x
    if method == "mad":
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-6
        return (x - med) / (1.4826 * mad)
    if method == "iqr":
        q1, q3 = np.percentile(x, [25, 75])
        iqr = (q3 - q1) + 1e-6
        med = np.median(x)
        return (x - med) / iqr
    raise ValueError(f"Unknown robust scaling method: {method}")

def _compute_off_threshold(flow_raw: np.ndarray) -> float:
    """OFF threshold is defined on RAW flow (positive)."""
    if flow_raw.size == 0:
        return 0.0

    if OFF_THRESH_METHOD == "percentile":
        base = float(np.percentile(flow_raw, OFF_THRESH_PERCENTILE))
        return max(1e-9, base)

    med = float(np.median(flow_raw))
    return max(1e-9, OFF_THRESH_FRAC_OF_MEDIAN * med)

def _longest_run(mask: np.ndarray) -> int:
    """Longest consecutive True run length."""
    best = 0
    cur = 0
    for v in mask:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best

def _band_limited_fft_peak(chunk: np.ndarray, fps: float) -> Tuple[float, float]:
    """
    Returns (dom_freq_hz, freq_power_ratio_band)
    - Hann window optional
    - search only within [FREQ_MIN_HZ, FREQ_MAX_HZ]
    """
    if chunk.size < 8:
        return 0.0, 0.0

    x = chunk.astype(np.float32)
    x = x - float(np.mean(x))

    if USE_HANN_WINDOW:
        x = x * np.hanning(len(x)).astype(np.float32)

    yf = np.abs(rfft(x))
    xf = rfftfreq(len(x), 1.0 / fps)

    band = (xf >= FREQ_MIN_HZ) & (xf <= FREQ_MAX_HZ)
    if not np.any(band):
        return 0.0, 0.0

    yf_band = yf[band]
    xf_band = xf[band]

    peak_local = int(np.argmax(yf_band))
    dom_freq = float(xf_band[peak_local])

    band_power = float(np.sum(yf_band) + 1e-6)
    peak_power = float(yf_band[peak_local])
    ratio = peak_power / band_power

    return dom_freq, float(ratio)

def _windowed_dataframe(
    flow_raw: np.ndarray,
    fps: float,
    window_sec: float,
    step_sec: float,
    enable_peak_sanitycheck: bool,
    diff_signal: Optional[np.ndarray],
) -> pd.DataFrame:
    """
    Create a per-window DataFrame. Pipeline will summarize this into per-video features.
    """
    win = int(round(window_sec * fps))
    step = int(round(step_sec * fps))

    if win < 8 or len(flow_raw) < win + 1:
        return pd.DataFrame()

    off_thresh = _compute_off_threshold(flow_raw)

    rows = []
    for start in range(0, len(flow_raw) - win + 1, step):
        end = start + win

        raw = flow_raw[start:end]
        mean_raw = float(np.mean(raw))
        std_raw = float(np.std(raw))
        jitter_cv_raw = float(std_raw / (mean_raw + 1e-6))

        off_mask = (raw < off_thresh)
        duty_cycle = float(1.0 - np.mean(off_mask))
        longest_off_s = float(_longest_run(off_mask) / fps)

        if mean_raw <= off_thresh or duty_cycle < 0.5:
            dom_freq = 0.0
            freq_power_ratio_band = 0.0
        else:
            dom_freq, freq_power_ratio_band = _band_limited_fft_peak(raw, fps)

        peak_freq = np.nan
        if enable_peak_sanitycheck and diff_signal is not None and len(diff_signal) >= end:
            dchunk = diff_signal[start:end]
            if np.std(dchunk) > 1e-6:
                min_dist = max(1, int(round(fps / float(PEAK_MIN_DIST_HZ))))
                prom = np.std(dchunk) * PEAK_MIN_PROM_FRAC
                peaks, _ = find_peaks(dchunk, distance=min_dist, prominence=prom)
                if len(peaks) > 2:
                    duration = (peaks[-1] - peaks[0]) / fps
                    if duration > 1e-6:
                        peak_freq = float((len(peaks) - 1) / duration)

        rows.append({
            "t_start_s": float(start / fps),
            "t_mid_s": float((start + win / 2) / fps),

            "mean_intensity_raw": mean_raw,
            "std_intensity_raw": std_raw,
            "jitter_cv_raw": jitter_cv_raw,

            "off_thresh_raw": float(off_thresh),
            "duty_cycle": duty_cycle,
            "longest_off_s": longest_off_s,

            "dom_freq_hz": dom_freq,
            "freq_power_ratio_band": freq_power_ratio_band,

            "peak_freq_hz_sanity": peak_freq,
        })

    return pd.DataFrame(rows)


# ============================================================
# PUBLIC API (this is what main.py should call)
# ============================================================

def compute_flow_window_features(
    flow_raw: np.ndarray,
    *,
    fps: float,
    diff_signal: Optional[np.ndarray] = None,
    window_size_sec: float = WINDOW_SIZE_SEC,
    step_size_sec: float = STEP_SIZE_SEC,
    on_window_duty_thresh: float = ON_WINDOW_DUTY_THRESH,
    enable_peak_sanitycheck: bool = ENABLE_PEAK_SANITYCHECK_DEFAULT,
) -> Dict[str, float]:
    """
    Compute per-video aggregated window features from the optical-flow signal.

    Args:
        flow_raw: 1D optical-flow signal (positive). Best if you pass the pump-on segment.
        fps: frames per second.
        diff_signal: optional 1D diff-based signal (same length as flow_raw), used only for sanity peak freq.
        window_size_sec: window length for analysis.
        step_size_sec: step size for windowing.
        on_window_duty_thresh: windows above this duty are considered "ON" windows.
        enable_peak_sanitycheck: if True, compute peak freq sanity features when diff_signal is given.

    Returns:
        dict of aggregated features (floats), ready to merge into your feature dict.
        Feature names are prefixed with "flow_" to avoid confusion with your existing ones.
    """
    flow_raw = np.asarray(flow_raw, dtype=np.float32)

    df = _windowed_dataframe(
        flow_raw=flow_raw,
        fps=float(fps),
        window_sec=float(window_size_sec),
        step_sec=float(step_size_sec),
        enable_peak_sanitycheck=bool(enable_peak_sanitycheck),
        diff_signal=diff_signal,
    )

    # If not enough data, return zeros (stable pipeline)
    if df.empty:
        return {
            "flow_n_windows": 0.0,
            "flow_mean_intensity_raw_med": 0.0,
            "flow_mean_intensity_raw_iqr": 0.0,
            "flow_duty_cycle_med": 0.0,
            "flow_on_window_ratio": 0.0,
            "flow_longest_off_s_max_on": 0.0,
            "flow_dom_freq_med_on": 0.0,
            "flow_freq_power_ratio_band_med_on": 0.0,
            "flow_jitter_cv_raw_med_on": 0.0,
            "flow_peak_freq_sanity_med_on": 0.0,
        }

    on_df = df[df["duty_cycle"] >= float(on_window_duty_thresh)]
    on_ratio = float(len(on_df) / max(1, len(df)))

    # Helper to safely compute stats
    def _med(s: pd.Series) -> float:
        return float(np.median(s.values)) if len(s) else 0.0

    def _iqr(s: pd.Series) -> float:
        if len(s) == 0:
            return 0.0
        v = s.values
        return float(np.percentile(v, 75) - np.percentile(v, 25))

    features = {
        "flow_n_windows": float(len(df)),
        "flow_mean_intensity_raw_med": _med(df["mean_intensity_raw"]),
        "flow_mean_intensity_raw_iqr": _iqr(df["mean_intensity_raw"]),
        "flow_duty_cycle_med": _med(df["duty_cycle"]),
        "flow_on_window_ratio": on_ratio,
    }

    # ON-window stats (avoid start/stop boundary polluting)
    features.update({
        "flow_longest_off_s_max_on": float(np.max(on_df["longest_off_s"])) if len(on_df) else 0.0,
        "flow_dom_freq_med_on": _med(on_df["dom_freq_hz"]) if len(on_df) else 0.0,
        "flow_freq_power_ratio_band_med_on": _med(on_df["freq_power_ratio_band"]) if len(on_df) else 0.0,
        "flow_jitter_cv_raw_med_on": _med(on_df["jitter_cv_raw"]) if len(on_df) else 0.0,
    })

    # Optional sanity peak frequency
    if "peak_freq_hz_sanity" in df.columns and df["peak_freq_hz_sanity"].notna().any():
        sane_on = on_df["peak_freq_hz_sanity"].dropna() if len(on_df) else pd.Series([], dtype=float)
        features["flow_peak_freq_sanity_med_on"] = float(np.median(sane_on.values)) if len(sane_on) else 0.0
    else:
        features["flow_peak_freq_sanity_med_on"] = 0.0

    return features
