import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import joblib
from audio_features import extract_features

x, y = [], []
data_dir = "data/"
labels = os.listdir(data_dir)

for label in labels:
    folder = os.path.join(data_dir, label)
    if not os.path.isdir(folder):
        continue
    for file in os.listdir(folder):
        if file.endswith(".wav") or file.endswith(".mp3"):
            path = os.path.join(folder, file)
            try:
                features = extract_features(path)
                if features is not None:
                    x.append(features)
                    y.append(label)
            except Exception as e:
                print(f"Error processing {path}: {e}")

# Check if data was loaded
if len(x) == 0:
    raise ValueError("No valid audio features found. Make sure there are .wav or .mp3 files in data folders.")

x = np.array(x)
y = np.array(y)
print("Label distribution:", Counter(y))

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
clf.fit(x_train, y_train)

# Predict
y_pred = clf.predict(x_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\n Model Evaluation:")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1 Score     : {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save metrics to CSV
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score": [accuracy, precision, recall, f1]
})
metrics_df.to_csv("model_metrics.csv", index=False)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix.png")
plt.show()

# Save model
joblib.dump(clf, "scream_classifier.pkl")
print("Model saved to: scream_classifier.pkl")
