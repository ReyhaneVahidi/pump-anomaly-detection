"""
Main script for extracting pump video features (per file + summary).

Workflow:
1. Read video frames (full + ROI)
2. Compute motion (diff-based full + ROI)
3. Detect pump start/end (based on FULL-FRAME motion)
4. Compute optical-flow ROI signal (for windowed features)
5. Segment signals to pump-on region
6. Compute temporal, frequency, brightness, energy, peak features (existing)
7. Compute windowed optical-flow features (duty cycle, longest off, FFT dom freq, PEAK sanity freq)
8. Save per-file results to CSV
9. Write summary + logs
"""

from pathlib import Path
import os
import pandas as pd
import logging
from typing import Optional, Dict, Any, List
from tqdm import tqdm

from pump_feature_extraction.config import VIDEO_DIR, OUTPUT_CSV, FPS
from pump_feature_extraction.video_io import read_video_frames
from pump_feature_extraction.motion import (
    compute_motion_from_frames,
    compute_optical_flow_signal,
    detect_pump_start_end,
)

from pump_feature_extraction.features.temporal_features import compute_temporal_features
from pump_feature_extraction.features.frequency_features import compute_frequency_features
from pump_feature_extraction.features.brightness_features import compute_brightness_features
from pump_feature_extraction.features.energy_features import compute_energy_features
from pump_feature_extraction.features.peak_features import compute_peak_features
from pump_feature_extraction.features.flow_window_features import compute_flow_window_features
from pump_feature_extraction.feature_packaging import pack_filename_timestamp


def extract_features_from_video(path: str) -> Optional[Dict[str, Any]]:
    """Extract all relevant features from a single pump video."""
    try:
        full_frames, roi_frames, brightness, frame_count = read_video_frames(path)
    except Exception as e:
        logging.error(f"{path} -> {e}")
        tqdm.write(f"[ERROR] {os.path.basename(path)} -> {e}")
        return None

    if frame_count is None or frame_count <= 1:
        logging.warning(f"Not enough frames in {path}")
        tqdm.write(f"[WARN] Too few frames: {os.path.basename(path)}")
        return None

    # 1) Diff-based motion signals (existing, fast)
    full_motion_raw, roi_motion_raw, roi_motion_norm = compute_motion_from_frames(full_frames, roi_frames)

    # 2) Optical-flow signal on ROI (new, for duty/off/frequency window features)
    roi_flow = compute_optical_flow_signal(
        roi_frames,
        downscale=0.5,  # speed; set to 1.0 on PC if you prefer
    )

    # 3) Start/end detection should use FULL-frame motion
    pump_start, pump_end = detect_pump_start_end(roi_motion_raw)

    if pump_start is None or pump_end is None:
        logging.warning(f"Could not detect pump start/end for {path}")
        tqdm.write(f"[WARN] No start/end for {os.path.basename(path)}")
        return None

    pump_start = max(0, int(pump_start))
    pump_end = min(int(pump_end), frame_count - 1)

    if pump_end <= pump_start:
        logging.warning(f"Invalid pump segment for {path}: start={pump_start}, end={pump_end}")
        tqdm.write(f"[WARN] Invalid segment for {os.path.basename(path)}")
        return None

    # 4) Segment of interest (pump-on)
    segment_motion = roi_motion_norm[pump_start:pump_end + 1]
    segment_brightness = brightness[pump_start:pump_end + 1]
    segment_flow = roi_flow[pump_start:pump_end + 1]

    features: Dict[str, Any] = {
        "file": os.path.basename(path),
        "duration_s": frame_count / FPS,
        "pump_start_s": pump_start / FPS,
        "pump_end_s": pump_end / FPS,
        "pump_duration_s": (pump_end - pump_start) / FPS,
    }

    # 5) Existing feature sets (diff-based ROI motion + brightness)
    features.update(compute_temporal_features(segment_motion, fps=FPS))
    features.update(compute_frequency_features(segment_motion, fps=FPS))
    features.update(compute_brightness_features(segment_brightness))
    features.update(compute_energy_features(segment_motion))
    features.update(compute_peak_features(segment_motion))

    # 6) New windowed optical-flow features (includes PEAK sanity frequency)
    # Pass diff-based motion as diff_signal so you get peak frequency too.
    features.update(
        compute_flow_window_features(
            segment_flow,
            fps=FPS,
            diff_signal=segment_motion,
            enable_peak_sanitycheck=True,
        )
    )

    # 7) Timestamp packaging
    features.update(pack_filename_timestamp(path))

    return features


def main() -> None:
    """Main entry point: processes all videos in VIDEO_DIR and logs results."""
    video_dir: Path = Path(VIDEO_DIR)
    output_dir: Path = Path(OUTPUT_CSV).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path: Path = output_dir / "feature_extraction.log"
    summary_path: Path = output_dir / "feature_extraction_summary.csv"

    # Logging setup
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    extensions: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")
    files: List[Path] = sorted([f for f in video_dir.iterdir() if f.suffix.lower() in extensions])

    summary: List[Dict[str, str]] = []

    print(f"[INFO] Found {len(files)} videos in {video_dir}")
    logging.info(f"Starting feature extraction for {len(files)} videos")

    # Progress bar with unit
    for f in tqdm(files, desc="Extracting Features", total=len(files), unit="video"):
        try:
            feat = extract_features_from_video(str(f))
            if feat is not None:
                out_file: Path = output_dir / f"{f.stem}_features.csv"
                pd.DataFrame([feat]).to_csv(out_file, index=False)
                logging.info(f"SUCCESS: {f.name} -> {out_file.name}")
                summary.append({"file": f.name, "status": "success"})
            else:
                logging.warning(f"FAILED (no features): {f.name}")
                summary.append({"file": f.name, "status": "failed"})
        except Exception as e:
            logging.error(f"EXCEPTION: {f.name} -> {e}")
            summary.append({"file": f.name, "status": f"error: {e}"})

    # Write summary CSV
    pd.DataFrame(summary).to_csv(summary_path, index=False)

    # Print summary stats
    success = sum(1 for x in summary if x["status"] == "success")
    failed = len(summary) - success
    print(f"\n✅ Extraction complete: {success}/{len(files)} succeeded, {failed} failed.")
    print(f"🧾 Summary saved to: {summary_path}")
    print(f"📜 Detailed log: {log_path}")

    logging.info(f"Feature extraction complete. Summary saved to {summary_path}")
    logging.info(f"Failed: {failed} out of {len(files)}")


if __name__ == "__main__":
    main()
