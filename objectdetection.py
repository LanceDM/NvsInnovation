import torch

class ObjectDetection:
    def __init__(self, model_path="yolov5s.pt"):
        """Load YOLOv5 model for object detection."""
        self.model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)

    def detect_objects(self, frame):
        """Detect objects in the given frame."""
        results = self.model(frame)

        # Extract bounding boxes, labels, and class ids
        bbox = results.xywh[0].cpu().numpy()  # Get bounding boxes in format [x, y, width, height]
        labels = results.names  # Label names from YOLOv5 (e.g., 'person', 'car', etc.)
        class_ids = results.xywh[0][:, -1].cpu().numpy()  # Get class IDs (e.g., 'person' is class 0)

        # Only keep bounding boxes for class 0 (person)
        human_bbox = []
        human_labels = []
        for i, class_id in enumerate(class_ids):
            if class_id == 0:  # class 0 corresponds to 'person'
                human_bbox.append(bbox[i])
                human_labels.append(labels[int(class_id)])  # 'person'

        return human_bbox, human_labels, results
