"""
Train baseline unsupervised anomaly detection models 
(Isolation Forest as primary baseline).

Steps:
1. Load processed + scaled feature data
2. Load scaler + final feature list
3. Train Isolation Forest
4. Compute anomaly scores
5. Compute threshold (percentile-based)
6. Save model + threshold + diagnostics
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
import joblib
import json

# --------------------------------------
# Paths
# --------------------------------------
PROCESSED_FEATURE_PATH = Path(r"D:\Masterthesis\pump-anomaly-detection\data\processed\features_clean_scaled.csv")
SCALER_PATH = Path(r"D:\Masterthesis\pump-anomaly-detection\data\processed\scaler.pkl")

MODEL_OUTPUT_PATH = Path(r"D:\Masterthesis\pump-anomaly-detection\models\isolation_forest.pkl")
THRESHOLD_OUTPUT_PATH = Path(r"D:\Masterthesis\pump-anomaly-detection\models\threshold.json")
SCORE_OUTPUT_PATH = Path(r"D:\Masterthesis\pump-anomaly-detection\models\training_scores.csv")

# --------------------------------------
# Load data
# --------------------------------------
def load_data():
    print("➡ Loading processed scaled features...")
    df = pd.read_csv(PROCESSED_FEATURE_PATH)

    print("➡ Loading scaler + feature list...")
    scaler_data = joblib.load(SCALER_PATH)
    feature_list = scaler_data["features"]

    # Only keep the final features used in preprocessing
    df_sel = df[feature_list].copy()

    return df, df_sel, feature_list

# --------------------------------------
# Train Isolation Forest
# --------------------------------------
def train_isolation_forest(X: pd.DataFrame) -> IsolationForest:
    print("➡ Training Isolation Forest baseline model...")
    
    model = IsolationForest(
        n_estimators=300, # number of trees
        contamination="auto",  # the proportion of outliers in the data set
        max_samples="auto", # number of samples to draw from X to train each base estimator
        random_state=42, #  for reproducible results across multiple function calls
        bootstrap=False, # If True, individual trees are fit on random subsets of the training data sampled with replacement.
        n_jobs=-1 # Use all available cores
    )

    model.fit(X)
    return model

# --------------------------------------
# Compute anomaly scores & threshold
# --------------------------------------
def compute_scores_and_threshold(model: IsolationForest, X: pd.DataFrame):
    print("➡ Computing anomaly scores...")

    # IsolationForest gives negative_outlier_factor_ after fit, but safer to use decision_function.
    scores = -model.decision_function(X)  # Higher = more abnormal

    # Choose a percentile for anomaly threshold:
    # 99.5% → only extreme 0.5% considered anomaly
    threshold = float(np.percentile(scores, 99.5))

    print(f"📌 Suggested threshold (99.5th percentile): {threshold:.5f}")

    return scores, threshold

# --------------------------------------
# Save everything
# --------------------------------------
def save_outputs(model, scores, threshold, df_full):
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("➡ Saving Isolation Forest model...")
    joblib.dump(model, MODEL_OUTPUT_PATH) # Save the trained model to disk as a pickle file

    print("➡ Saving threshold...")
    with open(THRESHOLD_OUTPUT_PATH, "w") as f:
        json.dump({"threshold": threshold}, f, indent=4) # Save threshold as JSON

    print("➡ Saving training score diagnostics...")
    out_df = df_full.copy()
    out_df["anomaly_score"] = scores # Add scores to full dataframe
    out_df.to_csv(SCORE_OUTPUT_PATH, index=False)

    print(f"✅ Model saved: {MODEL_OUTPUT_PATH}")
    print(f"✅ Threshold saved: {THRESHOLD_OUTPUT_PATH}")
    print(f"✅ Score file saved: {SCORE_OUTPUT_PATH}")

# --------------------------------------
# Main pipeline
# --------------------------------------
def run_training():
    df_full, df_features, feature_list = load_data()

    model = train_isolation_forest(df_features)
    scores, threshold = compute_scores_and_threshold(model, df_features)

    save_outputs(model, scores, threshold, df_full)

    print("\n🎉 Baseline model training completed!")

if __name__ == "__main__":
    run_training()
