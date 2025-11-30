"""
Main script for extracting pump video features.

Workflow:
1. Read video frames (full + ROI)
2. Compute motion
3. Detect pump start/end
4. Compute temporal, frequency, and brightness features
5. Pack timestamp info
6. Save results to CSV
"""

from pathlib import Path
import os
from typing import Optional, Dict, Any, List
import pandas as pd

from .config import VIDEO_DIR, OUTPUT_CSV, FPS
from .video_io import read_video_frames
from .motion import compute_motion_from_frames, detect_pump_start_end
from .features.temporal_features import compute_temporal_features
from .features.frequency_features import compute_frequency_features
from .features.brightness_features import compute_brightness_features
from .feature_packaging import pack_filename_timestamp
from .features.energy_features import compute_energy_features
from features.peak_features import compute_peak_features




def extract_features_from_video(path: str) -> Optional[Dict[str, Any]]:
    """
    Extract all relevant features from a single pump video.

    Args:
        path (str): Path to the video file.

    Returns:
        Optional[Dict[str, Any]]: Dictionary of features if extraction succeeds, else None.
    """
    try:
        full_frames, roi_frames, brightness, frame_count = read_video_frames(path)
    except Exception as e:
        print(f"[ERROR] {path} -> {e}")
        return None

    full_motion, roi_motion = compute_motion_from_frames(full_frames, roi_frames)
    pump_start, pump_end = detect_pump_start_end(full_motion)

    if pump_start is None or pump_end is None:
        print(f"[WARN] Could not detect pump start/end for {path}")
        return None

    # Extract segment of interest
    segment_motion = full_motion[pump_start:pump_end + 1]
    segment_brightness = brightness[pump_start:pump_end + 1]

    features: Dict[str, Any] = {
        "file": os.path.basename(path),
        "duration_s": frame_count / FPS,
        "pump_start_s": pump_start / FPS,
        "pump_end_s": pump_end / FPS,
        "pump_duration_s": (pump_end - pump_start) / FPS
    }

    # Compute features using modular functions
    features.update(compute_temporal_features(segment_motion, fps=FPS))
    features.update(compute_frequency_features(segment_motion, fps=FPS))
    features.update(compute_brightness_features(segment_brightness))
    features.update(compute_energy_features(segment_motion))
    features.update(compute_peak_features(segment_motion))

    # Add timestamp metadata
    features.update(pack_filename_timestamp(path))

    return features


def main() -> None:
    """
    Main entry point: processes all videos in VIDEO_DIR and saves features to OUTPUT_CSV.
    """
    video_dir = Path(VIDEO_DIR)
    rows: List[Dict[str, Any]] = []
    extensions = (".mp4", ".avi", ".mov", ".mkv")
    files = sorted([f for f in video_dir.iterdir() if f.suffix.lower() in extensions])

    for f in files:
        print(f"[INFO] Processing {f.name} ...")
        feat = extract_features_from_video(str(f))
        if feat is not None:
            rows.append(feat)

    if rows:
        df = pd.DataFrame(rows)
        out_path = Path(OUTPUT_CSV)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[DONE] Saved to {out_path}")
    else:
        print("[DONE] No features extracted.")


if __name__ == "__main__":
    main()
