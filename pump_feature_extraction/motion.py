"""
Motion computation functions for pump videos.

Includes:
- Frame-difference motion per frame (full + ROI)  [cheap, good for start/stop + gross events]
- Optical-flow motion signal for ROI              [better for rhythm/frequency features]
- Pump start/end detection (based on full-frame motion)

Design notes:
- We keep your existing diff-count motion because it is fast and robust.
- We add an optical-flow ROI signal (1D) that is much more suitable for dominant frequency,
  duty cycle, longest-off, etc. (windowed features live elsewhere).
"""

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
import cv2

from .config import MOTION_ABS_DIFF_THRESH, MOTION_MIN_PIXELS, PERSIST_FRAMES


# ============================================================
# Frame-difference motion (existing baseline)
# ============================================================

def compute_motion_from_frames(
    full_frames: list[np.ndarray],
    roi_frames: list[np.ndarray],
    diff_thresh: int = MOTION_ABS_DIFF_THRESH,
    blur_ksize: int = 5,
    normalize_by_roi_area: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute motion per frame for full-frame and ROI using absdiff + threshold + count_nonzero.

    Returns:
        full_motion_raw: float32 array, changed pixels per frame (full frame)
        roi_motion_raw:  float32 array, changed pixels per frame (ROI)
        roi_motion_norm: float32 array, roi_motion_raw normalized by ROI area (fraction of ROI that changed)
                         If normalize_by_roi_area=False, roi_motion_norm == roi_motion_raw.
    Notes:
        - First element is 0 by convention (no previous frame).
        - Normalization makes motion comparable across different ROI sizes.
    """
    if not full_frames or not roi_frames:
        z = np.zeros((0,), dtype=np.float32)
        return z, z, z

    n = min(len(full_frames), len(roi_frames))
    if n < 2:
        z = np.zeros((n,), dtype=np.float32)
        return z, z, z

    if blur_ksize % 2 == 0:
        blur_ksize += 1

    full_motion_raw = np.zeros((n,), dtype=np.float32)
    roi_motion_raw = np.zeros((n,), dtype=np.float32)

    prev_full = full_frames[0]
    prev_roi = roi_frames[0]

    for i in range(1, n):
        cur_full = full_frames[i]
        cur_roi = roi_frames[i]

        # Full-frame motion (raw pixels)
        diff_full = cv2.absdiff(prev_full, cur_full)
        _, bin_full = cv2.threshold(diff_full, diff_thresh, 255, cv2.THRESH_BINARY)
        bin_full = cv2.medianBlur(bin_full, blur_ksize)
        full_motion_raw[i] = float(np.count_nonzero(bin_full))

        # ROI motion (raw pixels)
        diff_roi = cv2.absdiff(prev_roi, cur_roi)
        _, bin_roi = cv2.threshold(diff_roi, diff_thresh, 255, cv2.THRESH_BINARY)
        bin_roi = cv2.medianBlur(bin_roi, blur_ksize)
        roi_motion_raw[i] = float(np.count_nonzero(bin_roi))

        prev_full = cur_full
        prev_roi = cur_roi

    # ROI area normalization (fraction of ROI moved)
    if normalize_by_roi_area:
        roi_h, roi_w = roi_frames[0].shape[:2] # 0 : choose from the first frame, all should be same size. 
        roi_area = float(roi_h * roi_w)
        roi_motion_norm = roi_motion_raw / (roi_area + 1e-9)
    else:
        roi_motion_norm = roi_motion_raw.copy()

    return full_motion_raw, roi_motion_raw, roi_motion_norm



# ============================================================
# Optical flow signal (NEW)
# ============================================================

def compute_optical_flow_signal(
    roi_frames: list[np.ndarray],
    *,
    clip_mag_pct: float = 99.0,
    normalize_by_roi_diag: bool = True,
    downscale: float = 1.0,
    farneback_params: Optional[dict] = None,
) -> np.ndarray:
    """
    Compute a 1D motion signal from ROI frames using Farneback optical flow.

    Output is per-frame-transition mean flow magnitude (positive).
    This signal is good for:
      - dominant frequency (FFT)
      - rhythm stability
      - duty cycle / longest off (after thresholding)

    Args:
        roi_frames: list of blurred grayscale ROI frames (uint8).
        clip_mag_pct: clip per-frame magnitude map at this percentile to reduce spikes/noise.
        normalize_by_roi_diag: if True, divide mean magnitude by ROI diagonal so zoom/ROI size changes
                               affect the signal less.
        downscale: optional speed-up factor (<1 reduces resolution). Use 0.5 on Pi if needed.
        farneback_params: optional dict to override Farneback settings.

    Returns:
        flow_signal: float32 array length = len(roi_frames), first element = 0.
    """
    if not roi_frames:
        return np.zeros((0,), dtype=np.float32)
    if len(roi_frames) < 2:
        return np.zeros((len(roi_frames),), dtype=np.float32)

    # Default Farneback settings (reasonable)
    fb = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=20,
        iterations=3,
        poly_n=5,
        poly_sigma=1.7,
        flags=0,
    )
    if farneback_params:
        fb.update(farneback_params)

    n = len(roi_frames)
    flow_signal = np.zeros((n,), dtype=np.float32)

    prev = roi_frames[0]

    # Optional downscale (speeds up; keep aspect)
    def _maybe_resize(gray: np.ndarray) -> np.ndarray:
        if downscale is None or downscale >= 0.999:
            return gray
        h, w = gray.shape[:2]
        nh = max(8, int(round(h * downscale)))
        nw = max(8, int(round(w * downscale)))
        return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)

    prev = _maybe_resize(prev)

    # ROI diag normalization (based on ROI frame size)
    if normalize_by_roi_diag:
        h0, w0 = prev.shape[:2]
        roi_diag = float(np.sqrt(h0 * h0 + w0 * w0) + 1e-6)
    else:
        roi_diag = 1.0

    for i in range(1, n):
        cur = _maybe_resize(roi_frames[i])

        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, **fb)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        if mag.size > 0 and clip_mag_pct is not None:
            hi = np.percentile(mag, clip_mag_pct)
            mag = np.clip(mag, 0, hi)

        flow_signal[i] = float(np.mean(mag) / roi_diag)

        prev = cur

    return flow_signal


# ============================================================
# Pump start/end detection (full-motion based)
# ============================================================

def detect_pump_start_end(
    full_motion: np.ndarray,
    min_pixels: int = MOTION_MIN_PIXELS,
    persist_frames: int = PERSIST_FRAMES
) -> Tuple[Optional[int], Optional[int]]:
    """
    Detect pump start and end frames based on full-frame motion.

    Logic:
    - Start: first index where motion >= min_pixels for persist_frames consecutive frames
    - End: last frame where motion >= min_pixels

    Args:
        full_motion: 1D array of motion values from compute_motion_from_frames(full_frames,...).
        min_pixels: minimum motion pixels to consider pump active.
        persist_frames: consecutive frames required to confirm pump start.

    Returns:
        (pump_start_frame, pump_end_frame), each Optional[int]
    """
    if full_motion is None or len(full_motion) == 0:
        return None, None

    if persist_frames < 1:
        persist_frames = 1

    pump_start: Optional[int] = None
    streak = 0

    for i, val in enumerate(full_motion):
        if val >= min_pixels:
            streak += 1
            if streak >= persist_frames and pump_start is None:
                pump_start = i - persist_frames + 1
        else:
            streak = 0

    pump_end: Optional[int] = None
    for i in range(len(full_motion) - 1, -1, -1):
        if full_motion[i] >= min_pixels:
            pump_end = i
            break

    return pump_start, pump_end
