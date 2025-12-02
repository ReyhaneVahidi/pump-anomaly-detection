"""
Preprocessing utilities for pump anomaly detection feature data.

Handles:
1. Loading raw feature CSVs
2. Cleaning missing/invalid values
3. Removing constant/quasi-constant features (safely using threshold=0.0)
4. Clipping outliers (numeric only)
5. Scaling numeric features
6. Saving cleaned & scaled DataFrames, scaler object, and the final feature list.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import joblib
from typing import List, Tuple, Dict, Any, Optional

# -----------------------------
# Paths
# -----------------------------
RAW_FEATURE_PATH = Path(r"D:\Masterthesis\pump-anomaly-detection\data\raw\normal\new_feature_set.csv")
OUTPUT_PROCESSED_PATH = Path(r"D:\Masterthesis\pump-anomaly-detection\data\processed\features_clean_scaled.csv")
SCALER_PATH = Path(r"D:\Masterthesis\pump-anomaly-detection\data\processed\scaler.pkl")

# -----------------------------
# Numeric features for modeling
# -----------------------------
NUMERIC_FEATURES = [
    'duration_s', 'pump_duration_s', 'motion_mean', 'motion_std', 'motion_max',
    'motion_sum', 'motion_first', 'motion_last', 'motion_skewness', 'motion_kurtosis',
    'motion_rms', 'motion_peak_count', 'motion_slope_start', 'motion_slope_end',
    'motion_slope_start_norm', 'motion_slope_end_norm', 'motion_median', 'motion_iqr',
    'motion_mean_abs_diff', 'motion_roll_slope_mean', 'motion_roll_slope_std', 
    'motion_roll_slope_max', 'motion_roll_slope_min', 'motion_roll_slope_last',
    'motion_autocorr_lag1', 'motion_autocorr_lag2', 'motion_autocorr_lag3',
    'motion_cv', 'motion_entropy', 'peak_interval_mean', 'start_var', 'end_var',
    'dominant_freq', 'spectral_centroid', 'spectral_bandwidth', 
    'brightness_mean', 'brightness_std', 'brightness_slope'
]

# -----------------------------
# Helper function to get current numeric features
# -----------------------------
def get_current_numeric_features(df: pd.DataFrame, initial_features: List[str]) -> List[str]:
    """Returns the intersection of initial_features and the current DataFrame columns."""
    return [col for col in initial_features if col in df.columns]

# -----------------------------
# Load & Clean
# -----------------------------
def load_feature_data(path: Path) -> pd.DataFrame:
    """Load raw feature CSV."""
    df = pd.read_csv(path)
    return df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf/-inf with NaN and fill NaNs in numeric features with median."""
    df = df.replace([np.inf, -np.inf], np.nan)
    # Use the original NUMERIC_FEATURES list for cleaning since no features are dropped yet
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].fillna(df[NUMERIC_FEATURES].median())
    return df

# -----------------------------
# Feature filtering
# -----------------------------
def remove_constant_features(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Remove constant/quasi-constant numeric features."""
    
    current_numeric_features = get_current_numeric_features(df, NUMERIC_FEATURES)
    
    # 1. Fit selector on only the numeric features
    selector = VarianceThreshold(threshold=threshold) # initialize the selector
    selector.fit(df[current_numeric_features]) # tells the selector to learn the necessary information from the data
    
    # 2. Get the list of features to keep
    kept_features: List[str] = df[current_numeric_features].columns[selector.get_support()].tolist()
    
    # 3. Identify and log dropped features (Enhancement 1)
    features_to_drop = [f for f in current_numeric_features if f not in kept_features]
    print(f"Dropped {len(features_to_drop)} constant/quasi-constant features: {features_to_drop}")
    
    # 4. Drop the columns that were removed
    df_filtered = df.drop(columns=features_to_drop)
    
    return df_filtered

def clip_outliers(df: pd.DataFrame, q: float = 0.995) -> pd.DataFrame:
    """Clip extreme outliers (Winsorization) on current numeric features only."""
    
    # Dynamically find the current numeric features
    current_numeric_features = get_current_numeric_features(df, NUMERIC_FEATURES)
    
    df_numeric = df[current_numeric_features].copy()
    upper = df_numeric.quantile(q)
    lower = df_numeric.quantile(1 - q)
    df_numeric = df_numeric.clip(lower=lower, upper=upper, axis=1)
    df[current_numeric_features] = df_numeric
    return df

# -----------------------------
# Scaling
# -----------------------------
def scale_features(df: pd.DataFrame, scaler: Optional[StandardScaler] = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """Scale current numeric features with StandardScaler."""
    
    # Dynamically find the current numeric features
    current_numeric_features = get_current_numeric_features(df, NUMERIC_FEATURES)
    
    if scaler is None:
        scaler = StandardScaler()
        df_scaled_arr = scaler.fit_transform(df[current_numeric_features])
    else:
        df_scaled_arr = scaler.transform(df[current_numeric_features])
        
    df_scaled = df.copy()
    df_scaled[current_numeric_features] = df_scaled_arr
    return df_scaled, scaler

# -----------------------------
# Pipeline
# -----------------------------
def run_preprocessing(
    input_path: Path = RAW_FEATURE_PATH,
    output_path: Path = OUTPUT_PROCESSED_PATH,
    scaler_path: Path = SCALER_PATH,
    variance_threshold: float = 0.0,
    clip_quantile: float = 0.995 # remove extreme 0.5% outliers
):
    """Full preprocessing pipeline."""
    print("➡ Loading feature data...")
    df = load_feature_data(input_path)

    print("➡ Cleaning data...")
    df = clean_dataframe(df)

    print("➡ Removing constant features...")
    df = remove_constant_features(df, threshold=variance_threshold)

    print("➡ Clipping outliers...")
    df = clip_outliers(df, q=clip_quantile)

    print("➡ Scaling numeric features...")
    df_scaled, scaler = scale_features(df)
    
    # Get the final list of features that were scaled
    final_features_list = get_current_numeric_features(df, NUMERIC_FEATURES)

    # Save outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_scaled.to_csv(output_path, index=False)
    
    # Save scaler and feature list together (Enhancement 2)
    scaler_data: Dict[str, Any] = {
        'scaler': scaler,
        'features': final_features_list
    }
    joblib.dump(scaler_data, scaler_path)
    
    print(f"✅ Preprocessing complete. Scaled features saved to {output_path}")
    print(f"✅ Scaler and feature list saved to {scaler_path}")


if __name__ == "__main__":
    run_preprocessing()