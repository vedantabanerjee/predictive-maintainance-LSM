# Cell 9: INFERENCE (file-level aggregation). Paste into VSCode or run here.
MODEL_PATH = "hybrid_model.keras"
SCALER_PATH = "scaler_1.pkl"
DATA_FOLDER = "."  # or path to folder with test files
DATA_COLUMNS = ["ax","ay","az","gx","gy","gz"]
ACC_CHANNELS = ["ax","ay","az"]
WINDOW_SIZE = 64
STRIDE = WINDOW_SIZE // 2
FS = 100
SPEC_NPERSEG = 32
SPEC_NOVERLAP = 16

import os, joblib, pandas as pd, numpy as np
from scipy.signal import spectrogram
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns

# set your test files mapping (filename -> true class index)
class_name = ["Motor OFF", "Motor On", "No Fan", "Bad Fan"]
file_class_map = {
    "TEST_motor_off.xlsx": 0,
    "TEST_motor_on_2.xlsx": 1,
    "TEST_motor_on_nofan_1.xlsx": 2,
    "TEST_motor_on_badfan_2.xlsx": 3
}

if not os.path.exists(MODEL_PATH):
    print("Model not found:", MODEL_PATH)
else:
    model = tf.keras.models.load_model(MODEL_PATH)
    if not os.path.exists(SCALER_PATH):
        print("Scaler not found:", SCALER_PATH)
    else:
        scaler = joblib.load(SCALER_PATH)

        y_true = []
        y_pred = []
        for filename, class_id in file_class_map.items():
            filepath = os.path.join(DATA_FOLDER, filename)
            if not os.path.exists(filepath):
                print("Missing test file:", filepath)
                continue
            try:
                if filename.lower().endswith(".xlxs"):
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_excel(filepath)
            except Exception as e:
                print("Error reading test file:", filename, e)
                continue

            if any(c not in df.columns for c in DATA_COLUMNS):
                print("Test file missing columns:", filename)
                continue

            arr = df[DATA_COLUMNS].values
            if arr.shape[0] < WINDOW_SIZE:
                print("Test file too short for any windows:", filename)
                continue

            scaled = scaler.transform(arr)
            windows = []
            specs = []
            for start in range(0, scaled.shape[0] - WINDOW_SIZE + 1, STRIDE):
                w = scaled[start:start+WINDOW_SIZE]
                windows.append(w)
                spec_chs = []
                for ch in ACC_CHANNELS:
                    idx = DATA_COLUMNS.index(ch)
                    f, tt, Sxx = spectrogram(w[:, idx], fs=FS, nperseg=SPEC_NPERSEG, noverlap=SPEC_NOVERLAP)
                    spec_chs.append(np.log1p(Sxx))
                spec_img = np.stack(spec_chs, axis=-1)
                specs.append(spec_img)

            if len(windows) == 0:
                print("No windows created for test file (too short):", filename)
                continue

            windows = np.stack(windows).astype(np.float32)
            specs = np.stack(specs).astype(np.float32)

            preds_proba = model.predict([windows, specs], verbose=0)
            avg_proba = preds_proba.mean(axis=0)       # file-level probability by averaging
            file_pred = int(np.argmax(avg_proba))

            # store
            y_true.append(class_id)
            y_pred.append(file_pred)
            print(f"{filename} -> pred: {file_pred}, true: {class_id}, probs: {np.round(avg_proba,3)}")

        # plot confusion matrix if any files predicted
        if len(y_true) > 0:
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_name))))
            plt.figure(figsize=(6,5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_name, yticklabels=class_name, cbar=False)
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("File-level confusion matrix")
            plt.show()
        else:
            print("No test predictions were produced.")
