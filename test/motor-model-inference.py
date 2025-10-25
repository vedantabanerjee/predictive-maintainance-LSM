import tensorflow as tf
import numpy as np
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

MODEL_PATH = "motor_model_1.keras"
SCALER_PATH = "scaler_1.pkl"
DATA_FOLDER = "."
DATA_COLUMNS = ["ax", "ay", "az", "gx", "gy", "gz"]

class_name = ["Motor OFF", "Motor On", "No Fan", "Bad Fan"]
file_class_map = {
    "TEST_motor_off.xlsx": 0,
    "TEST_motor_on_2.xlsx": 1,
    "TEST_motor_on_nofan_1.xlsx": 2,
    "TEST_motor_on_badfan_2.xlsx": 3
}

WINDOW_SIZE = 64
STRIDE = WINDOW_SIZE // 2


def make_windows(sensor_data, window_size, stride=1):
    num_samples = len(sensor_data)
    windows = []
    for start in range(0, num_samples - window_size + 1, stride):
        window = sensor_data[start:start + window_size]
        windows.append(window)
    return np.stack(windows)

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print(f"Model input shape: {model.input_shape}")

y_true = []
y_pred = []

# ---- Inference for all files ----
for filename, class_id in file_class_map.items():
    filepath = os.path.join(DATA_FOLDER, filename)
    sensor_data = pd.read_excel(filepath)[DATA_COLUMNS]
    scaled_data = scaler.transform(sensor_data)
    windows = make_windows(scaled_data, WINDOW_SIZE, STRIDE)

    preds = model.predict(windows, verbose=0)
    pred_classes = np.argmax(preds, axis=1)

    y_true.extend([class_id] * len(pred_classes))
    y_pred.extend(pred_classes)


cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_name,
    yticklabels=class_name,
    cbar=False
)

plt.title("Motor Health Prediction Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.tight_layout()
plt.show()
