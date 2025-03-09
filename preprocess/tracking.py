import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../sort")))

from sort import Sort  

import numpy as np

class Tracking:
    def __init__(self):
        """Initialize SORT tracker."""
        self.tracker = Sort()  # SORT tracker instance

    def update_tracks(self, detections):
        """Update the tracker with new detections.
        
        Args:
            detections (list of lists): List of detected bounding boxes from YOLOv5 in [x1, y1, x2, y2, conf] format.
        
        Returns:
            np.ndarray: Tracked objects in format [x1, y1, x2, y2, ID].
        """
        if len(detections) == 0:
            return np.empty((0, 5))  # Return empty if no detections
        
        detections = np.array(detections)  # Convert list to numpy array
        tracks = self.tracker.update(detections)  # Update tracker
        
        return tracks  # Each row: [x1, y1, x2, y2, ID]
