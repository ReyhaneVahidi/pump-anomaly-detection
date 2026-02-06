"""
Video I/O functions for reading frames, ROI extraction, and brightness computation.
"""

from typing import List, Tuple
import cv2
import numpy as np
import json
from pathlib import Path

from .config import GAUSSIAN_BLUR_K


# ------------------ ROI LOADING ------------------

ROI_JSON = Path(__file__).parents[1] / "roi" / "pump_rois.json"

with open(ROI_JSON, "r") as f:
    ROI_MAP = json.load(f)


def get_roi_for_video(video_path: str) -> Tuple[int, int, int, int]:
    """
    Return ROI as (x1, y1, x2, y2) for a given video.

    Assumes:
        screenshot filename == video stem + ".jpg"
        ROI stored as [x, y, w, h]
    """
    video_name = Path(video_path).stem
    image_name = f"{video_name}.jpg"

    if image_name not in ROI_MAP:
        raise KeyError(f"No ROI found for {image_name}")

    x, y, w, h = ROI_MAP[image_name]

    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid ROI size for {image_name}: {w}x{h}")

    return int(x), int(y), int(x + w), int(y + h)


# ------------------ VIDEO READING ------------------

def read_video_frames(
    path: str,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], int]:
    """
    Read a video file and return:
        - full-frame grayscale blurred frames
        - ROI grayscale blurred frames
        - mean brightness per frame
        - total frame count
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    full_frames: List[np.ndarray] = []
    roi_frames: List[np.ndarray] = []
    brightness: List[float] = []

    x1, y1, x2, y2 = get_roi_for_video(path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_K, 0)

        brightness.append(float(np.mean(gray_blur)))


        full_frames.append(gray_blur)
        roi_frames.append(gray_blur[y1:y2, x1:x2])

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return full_frames, roi_frames, brightness, frame_count
