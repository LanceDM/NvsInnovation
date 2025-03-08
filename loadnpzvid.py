import cv2
import numpy as np
from preprocess.preprocess import Preprocess

# Initialize Preprocessor
preprocessor = Preprocess(device="cpu")

# Load Video
video_path = "example.mp4"
cap = cv2.VideoCapture(video_path)

skeleton_data = []
tracked_ids = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame → Get skeletons & corresponding tracked IDs
    _, skeletons, ids = preprocessor.preprocess_frame(frame)

    skeleton_data.append(skeletons)
    
    # Ensure IDs match shape of skeletons (avoid mismatches)
    padded_ids = np.zeros((5,), dtype=int)  # Max 5 people, default 0
    padded_ids[:len(ids)] = ids[:5]  # Assign only up to 5 IDs
    tracked_ids.append(padded_ids)

cap.release()

# Convert to NumPy arrays & save NPZ
np.savez("skeleton_data.npz", data=np.array(skeleton_data), ids=np.array(tracked_ids))
print("✅ NPZ File Saved for ST-GCN Training!")
