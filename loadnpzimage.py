import numpy as np
import cv2

import sys
import os

# Add the preprocess folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "preprocess")))

from preprocess.preprocess import Preprocess

# Initialize Preprocessor
preprocessor = Preprocess(device="cpu")

# Load image
image_path = "image.png"
frame = cv2.imread(image_path)

# Process image → Get skeletons & corresponding tracked IDs
_, skeletons, tracked_ids = preprocessor.preprocess_frame(frame)

# Convert to NumPy arrays & save NPZ
np.savez("image_skeleton_data.npz", data=np.array([skeletons]), ids=np.array([tracked_ids]))
print("✅ NPZ File Saved for Image Processing!")
