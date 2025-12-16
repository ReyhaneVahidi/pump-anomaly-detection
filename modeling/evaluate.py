"""
Evaluate trained anomaly detection models for pump videos.

Steps:
1. Load processed & scaled features
2. Load trained model + scaler + feature list + threshold
3. Compute anomaly scores
4. Visualize score distribution
5. Flag potential anomalies (optional)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --------------------------------------
# Paths
# --------------------------------------
PROCESSED_FEATURE_PATH = Path(r"D:\Masterthesis\pump-anomaly-detection\data\processed\features_clean_scaled.csv")
SCALER_PATH = Path(r"D:\Masterthesis\pump-anomaly-detection\data\processed\scaler.pkl")
MODEL_PATH = Path(r"D:\Masterthesis\pump-anomaly-detection\models\isolation_forest.pkl")
THRESHOLD_PATH = Path(r"D:\Masterthesis\pump-anomaly-detection\models\threshold.json")
OUTPUT_SCORE_PATH = Path(r"D:\Masterthesis\pump-anomaly-detection\models\evaluation_scores.csv")
PLOT_PATH = Path(r"D:\Masterthesis\pump-anomaly-detection\models\anomaly_score_distribution.png")

# --------------------------------------
# Load data & model
# --------------------------------------
def load_data_and_model():
    print("➡ Loading processed feature data...")
    df = pd.read_csv(PROCESSED_FEATURE_PATH)

    print("➡ Loading scaler + feature list...")
    scaler_data = joblib.load(SCALER_PATH)
    feature_list = scaler_data["features"]
    scaler: StandardScaler = scaler_data["scaler"]

    print("➡ Loading trained model...")
    model = joblib.load(MODEL_PATH)

    print("➡ Loading threshold...")
    with open(THRESHOLD_PATH, "r") as f:
        threshold_data = json.load(f)
    threshold = threshold_data["threshold"]

    # Select only final features
    df_features = df[feature_list].copy()
    return df, df_features, scaler, model, threshold

# --------------------------------------
# Compute anomaly scores
# --------------------------------------
def compute_anomaly_scores(model, X: pd.DataFrame):
    scores = -model.decision_function(X)  # Higher = more abnormal
    return scores

# --------------------------------------
# Visualization
# --------------------------------------
def plot_scores(scores: np.ndarray, threshold: float, output_path: Path):
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, color='skyblue', edgecolor='k')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold={threshold:.5f}')
    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.title("Isolation Forest Anomaly Score Distribution")
    plt.legend() # display Legend(key)
    plt.tight_layout() # automatically adjusts subplot params so that the subplot(s) fits in to the figure area
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200) # dpi: dots per inch, shows image resolution
    plt.close()
    print(f"📌 Score distribution plot saved: {output_path}")

# --------------------------------------
# Optional: PCA projection for visualization
# --------------------------------------
def plot_pca(X: pd.DataFrame, scores: np.ndarray, output_path: Path):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X) # apply dimensionality reduction on X

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=scores, cmap='coolwarm', s=50)
    plt.colorbar(scatter, label='Anomaly Score')
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA Projection Colored by Anomaly Score")
    plt.tight_layout()
    plt.savefig(output_path.with_name("pca_projection.png"), dpi=200)
    plt.close()
    print(f"📌 PCA projection plot saved: {output_path.with_name('pca_projection.png')}")

# --------------------------------------
# Main evaluation
# --------------------------------------
def run_evaluation():
    df_full, df_features, scaler, model, threshold = load_data_and_model()

    print("➡ Computing anomaly scores...")
    scores = compute_anomaly_scores(model, df_features)

    # Flag potential anomalies
    df_eval = df_full.copy()
    df_eval["anomaly_score"] = scores
    df_eval["is_anomaly"] = df_eval["anomaly_score"] > threshold

    # Save evaluation results
    OUTPUT_SCORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_eval.to_csv(OUTPUT_SCORE_PATH, index=False)
    print(f"✅ Evaluation scores saved to: {OUTPUT_SCORE_PATH}")

    # Plot distribution and PCA
    plot_scores(scores, threshold, PLOT_PATH)
    plot_pca(df_features, scores, PLOT_PATH)

    print("🎉 Evaluation completed!")

if __name__ == "__main__":
    run_evaluation()
