import cv2
from preprocess.objectdetection import ObjectDetection
from preprocess.skeletalmapping import SkeletalMapping


class Preprocess:
  def __init__(self, device="cpu"):
    """Initialize both object detection and skeletal mapping."""
    self.device = device
    self.object_detector = ObjectDetection(device=self.device)
    self.skeletal_mapper = SkeletalMapping()

  def preprocess_frame(self, frame):
    """Preprocess a frame by detecting objects and mapping skeletons on detected humans."""
    # Detect human objects (class 0 in YOLOv5)
    bbox, labels, results = self.object_detector.detect_objects(frame)

    for det in bbox:
        # Get bounding box coordinates
        x_center, y_center, width, height = det[:4]
        x1, y1 = int(x_center - width / 2), int(y_center - height / 2) 
        x2, y2 = int(x_center + width / 2), int(y_center + height / 2) 

        # Crop the detected human region from the frame
        cropped_frame = frame[y1:y2, x1:x2]

        # Apply skeletal mapping to the cropped frame
        skeleton_landmarks = self.skeletal_mapper.map_skeletons(cropped_frame)

        # If skeleton landmarks are detected, map them back to the original frame
        if skeleton_landmarks:
            for landmark in skeleton_landmarks.landmark:  
                # Adjust coordinates to map skeletal points back to the original frame
                x, y = int(landmark.x * width + x1), int(landmark.y * height + y1)
                # Draw skeletal points on the original frame
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  

    return frame
