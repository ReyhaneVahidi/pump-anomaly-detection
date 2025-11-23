"""
Motion computation functions for pump videos.

Includes:
- Motion per frame (full and ROI)
- Pump start and end detection
"""

from typing import Tuple
import numpy as np
import cv2
from .config import MOTION_ABS_DIFF_THRESH, MOTION_MIN_PIXELS, PERSIST_FRAMES
from .utils import compute_normalized_slopes, compute_slope


def compute_motion_from_frames(
    full_frames: list[np.ndarray],
    roi_frames: list[np.ndarray],
    diff_thresh: int = MOTION_ABS_DIFF_THRESH
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute motion per frame for full-frame and ROI.

    Args:
        full_frames (list[np.ndarray]): List of blurred grayscale full frames.
        roi_frames (list[np.ndarray]): List of blurred grayscale ROI frames.
        diff_thresh (int, optional): Threshold for pixel difference. Defaults to config value.

    Returns:
        Tuple[np.ndarray, np.ndarray]: full_motion, roi_motion arrays (counts of changed pixels per frame).
    """
    full_motion: list[float] = [0]
    roi_motion: list[float] = [0]

    prev_full = full_frames[0]
    prev_roi = roi_frames[0]

    for cur_full, cur_roi in zip(full_frames[1:], roi_frames[1:]):
        # full-frame motion
        diff_full = cv2.absdiff(prev_full, cur_full)
        _, bin_full = cv2.threshold(diff_full, diff_thresh, 255, cv2.THRESH_BINARY)
        bin_full = cv2.medianBlur(bin_full, 5)
        full_motion.append(float(np.count_nonzero(bin_full)))

        # ROI motion
        diff_roi = cv2.absdiff(prev_roi, cur_roi)
        _, bin_roi = cv2.threshold(diff_roi, diff_thresh, 255, cv2.THRESH_BINARY)
        bin_roi = cv2.medianBlur(bin_roi, 5)
        roi_motion.append(float(np.count_nonzero(bin_roi)))

        prev_full = cur_full
        prev_roi = cur_roi

    return np.array(full_motion, dtype=float), np.array(roi_motion, dtype=float)


def detect_pump_start_end(
    full_motion: np.ndarray,
    min_pixels: int = MOTION_MIN_PIXELS,
    persist_frames: int = PERSIST_FRAMES
) -> Tuple[int | None, int | None]:
    """
    Detect pump start and end frames based on full-frame motion.

    Args:
        full_motion (np.ndarray): Array of full-frame motion values.
        min_pixels (int, optional): Minimum motion pixels to consider pump active.
        persist_frames (int, optional): Number of consecutive frames to confirm motion.

    Returns:
        Tuple[int | None, int | None]: pump_start_frame, pump_end_frame (None if not detected)
    """
    pump_start: int | None = None
    streak = 0
    for i, val in enumerate(full_motion):
        if val >= min_pixels:
            streak += 1
            if streak >= persist_frames and pump_start is None:
                pump_start = i - persist_frames + 1
        else:
            streak = 0

    last_motion_frame: int | None = None
    for i in range(len(full_motion) - 1, -1, -1):
        if full_motion[i] >= min_pixels:
            last_motion_frame = i
            break

    return pump_start, last_motion_frame
