import cv2
import numpy as np

import sys
import os

# Add the preprocess folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "preprocess")))

from preprocess.preprocess import Preprocess
from preprocess.preprocess import Preprocess

# Initialize Preprocess class
preprocessor = Preprocess(device="cpu")

# Load image
image_path = "image.png"
frame = cv2.imread(image_path)

# Process the image
processed_frame, skeletons, tracked_ids = preprocessor.preprocess_frame(frame)

# Draw bounding boxes and skeletons
for i, skeleton in enumerate(skeletons):
    for (x, y) in skeleton:
        cv2.circle(processed_frame, (int(x), int(y)), 3, (0, 255, 0), -1)

# Show image with detections
cv2.imshow("Image Testing", processed_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
