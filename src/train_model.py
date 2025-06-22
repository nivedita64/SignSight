import pandas as pd
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from tqdm import tqdm

CSV_PATH = "../dataset/asl_combined_AtoL.csv"
MODEL_PATH = "../models/sign_rf_model.pkl"
IMG_SIZE = 64  # Resize images to 64x64

# Load CSV
df = pd.read_csv(CSV_PATH)

X = []
y = []

print("Loading and processing images...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    img = cv2.imread(row["image_path"])
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    X.append(img.flatten())
    y.append(row["label"])

X = np.array(X)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")