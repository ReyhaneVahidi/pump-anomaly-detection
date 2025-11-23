"""
Configuration settings for the pump feature extraction project.

This file centralizes constants such as:
- ROI coordinates
- Frame rate
- Threshold values
- Input/output directories
- Other relevant parameters
"""

from pathlib import Path
from typing import Tuple 


# --- VIDEO SETTINGS ---
FPS: int = 30  # Frames per second of the pump videos

# ROI defined as (x1, y1, x2, y2)
ROI: Tuple[int, int, int, int] = (724, 478, 957, 697)


# --- DIRECTORIES ---
# Folder containing input videos
VIDEO_DIR: Path = Path(r"D:\Masterthesis\pi videos\videos")

# Folder where extracted feature CSV will be saved
OUTPUT_CSV: Path = Path(r"D:\Masterthesis\pi videos\results\new_feature_set.csv")

# === MOTION DETECTION PARAMETERS ===
MOTION_ABS_DIFF_THRESH: int = 15  # Threshold for pixel difference to count as motion
MOTION_MIN_PIXELS: int = 1000     # Minimum pixels changed to consider pump active
PERSIST_FRAMES: int = 6           # Number of consecutive frames above threshold to confirm motion

# === OTHER PARAMETERS ===
GAUSSIAN_BLUR_K: Tuple[int, int] = (5, 5)  # Kernel size for Gaussian blur
