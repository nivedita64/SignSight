import cv2
import joblib
import numpy as np
import pyttsx3

MODEL_PATH = "../models/sign_rf_model.pkl"
IMG_SIZE = 64

# Load the trained model
clf = joblib.load(MODEL_PATH)

# Initialize text-to-speech engine
engine = pyttsx3.init()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not detected! Please check your camera connection.")
    exit()
print("✅ Webcam connected. Press 'q' to quit.")

last_pred = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Center crop and resize to match training images
    h, w, _ = frame.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    crop = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_flat = img.flatten().reshape(1, -1)

    # Predict
    pred = clf.predict(img_flat)[0]

    # Speak only if prediction changes
    if pred != last_pred:
        engine.say(f"The sign is {pred}")
        engine.runAndWait()
        last_pred = pred

    # Show prediction
    cv2.putText(frame, f"Sign: {pred}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()