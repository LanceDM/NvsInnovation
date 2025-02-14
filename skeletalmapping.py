import mediapipe as mp
import cv2

class SkeletalMapping:
    def __init__(self):
        """Initialize the BlazePose model."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
    
    def map_skeletons(self, frame):
        """Map skeletons on the given frame using BlazePose."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
        results = self.pose.process(rgb_frame)  # Process frame with BlazePose
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)  # Draw skeleton landmarks
        
        return results.pose_landmarks  # Return landmarks if detected
