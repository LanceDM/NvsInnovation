import cv2
import sys
import os
import torch
# Add the preprocess folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "preprocess")))

from preprocess.preprocess import Preprocess
from preprocess.preprocess2 import Preprocess as preprocess2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} for processing.")

# Initialize Preprocess class
#preprocessor = preprocess2(device=device)
preprocessor = Preprocess(device=device)

# Open video file
video_path = "vidtesting.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    sys.exit()

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
