import mediapipe as mp
import cv2
import numpy as np

class SkeletalMapping:
    def __init__(self):
        """Initialize BlazePose for multi-person processing."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Select only 17 keypoints (COCO format)
        self.selected_keypoints = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    def map_skeletons(self, frame, bboxes):
        """Apply BlazePose on each detected person and return only selected keypoints."""
        skeletons = []

        for (x1, y1, x2, y2) in bboxes:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                continue

            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_crop)

            if results.pose_landmarks:
                # Convert keypoints from normalized (0-1) to absolute coordinates
                skeleton = np.array([[lm.x * person_crop.shape[1] + x1, lm.y * person_crop.shape[0] + y1]
                                     for i, lm in enumerate(results.pose_landmarks.landmark) if i in self.selected_keypoints])
                skeletons.append(skeleton)

        return skeletons  # List of (17, 2) NumPy arrays
