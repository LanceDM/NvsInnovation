import cv2
import numpy as np
from objectdetection import ObjectDetection
from skeletalmapping import SkeletalMapping
from tracking import Tracking  

class Preprocess:
    def __init__(self, device="cpu"):
        """Initialize object detection, skeletal mapping, and tracking."""
        self.device = device
        self.object_detector = ObjectDetection(device=self.device)
        self.skeletal_mapper = SkeletalMapping()
        self.tracker = Tracking()  

    def preprocess_frame(self, frame):
        """Preprocess a frame by detecting objects, tracking them, and mapping skeletons."""
        bbox, confs, results = self.object_detector.detect_objects(frame)
        
        # Format detections for SORT (x1, y1, x2, y2, conf)
        detections = [list(bbox[i]) + [confs[i]] for i in range(len(bbox))]
        
        # Update tracker with new detections
        tracked_objects = self.tracker.update_tracks(detections)
        
        skeleton_data = []
        tracked_ids = []

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)  # Get bounding box and ID
            tracked_ids.append(track_id)
            
            # Apply BlazePose 
            skeletons = self.skeletal_mapper.map_skeletons(frame, [(x1, y1, x2, y2)])
            
            for skeleton in skeletons:
                skeleton_data.append((track_id, skeleton))  # Store ID with skeleton

        # Ensure fixed number of people per frame (for ST-GCN)
        max_people = 5
        max_keypoints = 17  # Reduced keypoints for COCO compatibility
        processed_skeleton_data = np.zeros((max_people, max_keypoints, 2))  # Default to 0 (no detection)
        processed_ids = np.zeros((max_people,), dtype=int)  # Default IDs to 0

        for i, (track_id, skeleton) in enumerate(skeleton_data[:max_people]):  
            processed_skeleton_data[i, :len(skeleton), :] = skeleton  # Assign skeleton data
            processed_ids[i] = track_id  # Store tracking ID

        # Draw bounding boxes and skeletons on the frame
        self.draw_bounding_boxes(frame, tracked_objects)
        self.draw_skeletons(frame, skeleton_data)

        return frame, processed_skeleton_data, processed_ids

    def draw_bounding_boxes(self, frame, tracked_objects):
        """Draw bounding boxes with tracking IDs on the frame."""
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box
            cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def draw_skeletons(self, frame, skeleton_data):
        """Draw skeletons on the frame."""
        for track_id, skeleton in skeleton_data:
            for (x, y) in skeleton:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)  # Green keypoints

    def save_to_npz(self, file_path, video_name, skeleton_data, tracked_ids):
        """Save skeleton data and tracked IDs to an NPZ file for ST-GCN training."""
        np.savez(file_path, data=skeleton_data, ids=tracked_ids, video_name=video_name)
        print(f"Saved NPZ file: {file_path}")
