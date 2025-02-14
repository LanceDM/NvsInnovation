from objectdetection import ObjectDetection
from skeletalmapping import SkeletalMapping
import cv2

class Preprocess:
    def __init__(self):
        """Initialize both object detection and skeletal mapping."""
        self.object_detector = ObjectDetection()
        self.skeletal_mapper = SkeletalMapping()

    def preprocess_frame(self, frame):
        """Preprocess a frame by detecting objects and mapping skeletons."""
        # Object detection (only human detection)
        bbox, labels, results = self.object_detector.detect_objects(frame)

        # Filter detections for humans (class 0 in YOLOv5)
        for det in bbox:
            x_center, y_center, width, height = det[:4]  # Get bounding box coordinates
            x1 = int((x_center - width / 2))  # Calculate top-left x coordinate
            y1 = int((y_center - height / 2))  # Calculate top-left y coordinate
            x2 = int((x_center + width / 2))  # Calculate bottom-right x coordinate
            y2 = int((y_center + height / 2))  # Calculate bottom-right y coordinate

            # Draw bounding box for detected humans (class 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue color, thickness = 2

            # Add label (if needed)
            label = "Person"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Skeletal mapping
        skeletons = self.skeletal_mapper.map_skeletons(frame)

        return results, skeletons  # Return both detection and skeleton results
