"""
Main script for extracting pump video features (per file + summary).

Workflow:
1. Read video frames (full + ROI)
2. Compute motion
3. Detect pump start/end
4. Compute temporal, frequency, and brightness features
5. Save per-file results to CSV
6. Write summary + logs
"""

from pathlib import Path
import os
import pandas as pd
import logging
from typing import Optional, Dict, Any, List
from tqdm import tqdm

from pump_feature_extraction.config import VIDEO_DIR, OUTPUT_CSV, FPS
from pump_feature_extraction.video_io import read_video_frames
from pump_feature_extraction.motion import compute_motion_from_frames, detect_pump_start_end
from pump_feature_extraction.features.temporal_features import compute_temporal_features
from pump_feature_extraction.features.frequency_features import compute_frequency_features
from pump_feature_extraction.features.brightness_features import compute_brightness_features
from pump_feature_extraction.features.energy_features import compute_energy_features
from pump_feature_extraction.features.peak_features import compute_peak_features
from pump_feature_extraction.feature_packaging import pack_filename_timestamp


def extract_features_from_video(path: str) -> Optional[Dict[str, Any]]:
    """Extract all relevant features from a single pump video."""
    try:
        full_frames, roi_frames, brightness, frame_count = read_video_frames(path)
    except Exception as e:
        logging.error(f"{path} -> {e}")
        tqdm.write(f"[ERROR] {os.path.basename(path)} -> {e}")
        return None

    full_motion, roi_motion = compute_motion_from_frames(full_frames, roi_frames)
    pump_start, pump_end = detect_pump_start_end(full_motion)

    if pump_start is None or pump_end is None:
        logging.warning(f"Could not detect pump start/end for {path}")
        tqdm.write(f"[WARN] No start/end for {os.path.basename(path)}")
        return None

    # Segment of interest
    segment_motion = full_motion[pump_start:pump_end + 1]
    segment_brightness = brightness[pump_start:pump_end + 1]

    features: Dict[str, Any] = {
        "file": os.path.basename(path),
        "duration_s": frame_count / FPS,
        "pump_start_s": pump_start / FPS,
        "pump_end_s": pump_end / FPS,
        "pump_duration_s": (pump_end - pump_start) / FPS,
    }

    # Compute feature sets
    features.update(compute_temporal_features(segment_motion, fps=FPS))
    features.update(compute_frequency_features(segment_motion, fps=FPS))
    features.update(compute_brightness_features(segment_brightness))
    features.update(compute_energy_features(segment_motion))
    features.update(compute_peak_features(segment_motion))
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
