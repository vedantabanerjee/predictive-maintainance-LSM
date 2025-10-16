import argparse
import json
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras
from pathlib import Path

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def load_excel_files(file_list, data_columns):
    """
    Load and concatenate multiple Excel files containing vibration data.
    Each file should have the same columns as used during training.
    """
    all_data = []
    for file_path in file_list:
        df = pd.read_excel(file_path)
        # Extract only the relevant sensor columns (ax, ay, az, gx, gy, gz)
        df = df[data_columns]
        all_data.append(df)
        print(f"Loaded {file_path} with shape {df.shape}")
    combined = pd.concat(all_data, ignore_index=True)
    print(f"Combined dataset shape: {combined.shape}")
    return combined


def make_windows(data, window_size, stride=1):
    """
    Split the data into overlapping windows of size `window_size`.
    Returns an array of shape (num_windows, window_size, num_channels).
    """
    num_samples = len(data)
    windows = []
    for start in range(0, num_samples - window_size + 1, stride):
        window = data[start:start + window_size]
        windows.append(window)
    windows = np.stack(windows)
    print(f"Created {windows.shape[0]} windows of shape ({window_size}, {data.shape[1]})")
    return windows


def print_topk_predictions(predictions, index_to_class, k=3):
    """
    Print top-k predicted classes and their probabilities for each window.
    """
    for i, probs in enumerate(predictions):
        top_indices = np.argsort(probs)[::-1][:k]
        top_scores = probs[top_indices]
        top_labels = [index_to_class.get(int(idx), str(idx)) for idx in top_indices]

        print(f"\nWindow {i}: Top-{k} predictions")
        print("-" * 40)
        for lab, sc in zip(top_labels, top_scores):
            print(f"  {str(lab):20s}  {sc:.4f}")  # <-- fix applied
    print("\nInference completed successfully.")


# ------------------------------------------------------------
# Main inference routine
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run inference on motor vibration data using trained model.")
    parser.add_argument("--model", type=str, required=True, help="Path to saved .keras model file.")
    parser.add_argument("--scaler", type=str, required=True, help="Path to saved scaler.pkl file.")
    parser.add_argument("--meta", type=str, required=True, help="Path to model_meta.json file.")
    parser.add_argument("--from-excel", nargs="+", required=True, help="Excel files to load for inference.")
    parser.add_argument("--window-size", type=int, default=100, help="Window size used during training.")
    parser.add_argument("--stride", type=int, default=50, help="Stride for windowing.")
    args = parser.parse_args()

    # --------------------------------------------------------
    # 1️⃣ Load the model, scaler, and metadata
    # --------------------------------------------------------
    print("\nLoading model and metadata...")
    model = keras.models.load_model(args.model)
    scaler = joblib.load(args.scaler)
    with open(args.meta, "r") as f:
        meta = json.load(f)

    data_columns = meta["data_columns"]

    #Handle both possible meta structures
    if "class_to_index" in meta:
        class_to_index = meta["class_to_index"]
        index_to_class = {v: k for k, v in class_to_index.items()}
    elif "classes" in meta:
        classes = meta["classes"]
        index_to_class = {i: name for i, name in enumerate(classes)}
    else:
        raise KeyError("Neither 'class_to_index' nor 'classes' found in model_meta.json")

    print(f"Data columns used: {data_columns}")
    print(f"Detected {len(index_to_class)} classes: {list(index_to_class.values())}")

    # --------------------------------------------------------
    # 2️⃣ Load and preprocess Excel data
    # --------------------------------------------------------
    data_df = load_excel_files(args.from_excel, data_columns)
    data_np = data_df.values.astype(np.float32)

    # Apply the same scaler used during training
    data_scaled = scaler.transform(data_np)
    print("Applied feature scaling successfully.")

    # --------------------------------------------------------
    # 3️⃣ Create windows
    # --------------------------------------------------------
    windows = make_windows(data_scaled, window_size=args.window_size, stride=args.stride)

    # --------------------------------------------------------
    # 4️⃣ Perform inference
    # --------------------------------------------------------
    print("\nRunning model prediction...")
    predictions = model.predict(windows, verbose=1)

    # --------------------------------------------------------
    # 5️⃣ Display results
    # --------------------------------------------------------
    print_topk_predictions(predictions, index_to_class, k=3)


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
