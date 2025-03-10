import sys
import os
import torch
# Add the preprocess folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "preprocess")))
from preprocess.preprocess import Preprocess
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for processing.")

preprocessor = Preprocess(device=device)
cap = cv2.VideoCapture(0)  # Use webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame, skeletons, ids = preprocessor.preprocess_frame(frame)

    # Display the frame
    cv2.imshow("Live Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
