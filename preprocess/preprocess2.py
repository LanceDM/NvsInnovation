import cv2
import numpy as np
import time
from objectdetection import ObjectDetection
from skeletalmapping import SkeletalMapping
from tracking import Tracking  # Import the tracking module

class Preprocess:
    def __init__(self, device="cpu", batch_size=4, target_fps=10):
        """Initialize object detection, skeletal mapping, tracking, and processing parameters."""
        self.device = device
        self.object_detector = ObjectDetection(device=self.device)
        self.skeletal_mapper = SkeletalMapping()
        self.tracker = Tracking()  # Initialize SORT tracker
        self.batch_size = batch_size  # Number of frames per batch
        self.target_fps = target_fps  # Target frames per second
        self.frame_interval = 1.0 / target_fps
        self.last_frame_time = time.time()
        self.frame_batch = []  # Buffer for batching frames

    def preprocess_frame(self, frame):
        """Preprocess a frame by detecting objects, tracking them, and mapping skeletons."""
        # **Frame Rate Limiting**
        current_time = time.time()
        if current_time - self.last_frame_time < self.frame_interval:
            return None, None, None  # Skip processing to maintain FPS
        self.last_frame_time = current_time

        self.frame_batch.append(frame)
        if len(self.frame_batch) < self.batch_size:
            return None, None, None  # Wait until batch is full

        processed_batch = []
        processed_ids_batch = []

        for batch_frame in self.frame_batch:
            bbox, confs, results = self.object_detector.detect_objects(batch_frame)
            detections = [list(bbox[i]) + [confs[i]] for i in range(len(bbox))]
            tracked_objects = self.tracker.update_tracks(detections)
            skeleton_data = []
            tracked_ids = []

            for obj in tracked_objects:
                x1, y1, x2, y2, track_id = map(int, obj)
                tracked_ids.append(track_id)
                skeletons = self.skeletal_mapper.map_skeletons(batch_frame, [(x1, y1, x2, y2)])
                for skeleton in skeletons:
                    skeleton_data.append((track_id, skeleton))

            max_people = 5
            max_keypoints = 17  # Reduced keypoints for COCO compatibility
            processed_skeleton_data = np.zeros((max_people, max_keypoints, 2))
            processed_ids = np.zeros((max_people,), dtype=int)

            for i, (track_id, skeleton) in enumerate(skeleton_data[:max_people]):  
                processed_skeleton_data[i, :len(skeleton), :] = skeleton  # Assign skeleton data
                processed_ids[i] = track_id  # Store tracking ID

            # Draw bounding boxes and skeletons
            self.draw_bounding_boxes(batch_frame, tracked_objects)
            self.draw_skeletons(batch_frame, skeleton_data)

            processed_batch.append(processed_skeleton_data)
            processed_ids_batch.append(processed_ids)

        self.frame_batch = []  # Clear batch buffer
        return frame, np.array(processed_batch), np.array(processed_ids_batch)

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
