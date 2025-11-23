"""
Video I/O functions for reading frames, ROI, and brightness.
"""

from typing import List, Tuple
import cv2
import numpy as np
from .config import ROI, GAUSSIAN_BLUR_K

def read_video_frames(path: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], int]:
    """
    Read a video file and return full-frame and ROI frames, along with brightness per frame.

    Args:
        path (str): Path to the video file.

    Returns:
        Tuple containing:
            - full_frames (List[np.ndarray]): List of blurred grayscale full frames.
            - roi_frames (List[np.ndarray]): List of blurred grayscale ROI frames.
            - brightness_list (List[float]): Mean brightness per frame.
            - frame_count (int): Total number of frames in the video.

    Raises:
        IOError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    full_frames: List[np.ndarray] = []
    roi_frames: List[np.ndarray] = []
    brightness: List[float] = []

    x1, y1, x2, y2 = ROI

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_K, 0)

        # Record brightness
        brightness.append(float(np.mean(gray_blur)))

        # Store frames
        full_frames.append(gray_blur)
        roi_frames.append(gray_blur[y1:y2, x1:x2])

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return full_frames, roi_frames, brightness, frame_count
