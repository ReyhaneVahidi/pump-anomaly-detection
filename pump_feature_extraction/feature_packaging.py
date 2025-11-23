"""
Extract timestamp information from video filename.
"""

from typing import Dict, Any
import os
from datetime import datetime


def pack_filename_timestamp(path: str) -> Dict[str, Any]:
    """
    Extract timestamp from filename in format YYYYMMDD_HHMMSS.

    Args:
        path (str): Path to video file.

    Returns:
        Dict[str, Any]: Dictionary with year, month, day, hour, minute, weekday.
                        If parsing fails, values are None.
    """
    features: Dict[str, Any] = {}
    try:
        ts = os.path.basename(path).split('.')[0]
        dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
        features.update({
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "hour": dt.hour,
            "minute": dt.minute,
            "weekday": dt.weekday()
        })
    except Exception:
        features.update({
            "year": None,
            "month": None,
            "day": None,
            "hour": None,
            "minute": None,
            "weekday": None
        })
    return features
