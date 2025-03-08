import torch

class ObjectDetection:
    def __init__(self, model_path="yolov5s", device="cpu"):
        """Load YOLOv5 model for object detection."""
        self.device = device
        self.model = torch.hub.load("ultralytics/yolov5", model_path, pretrained=True, device=self.device)

    def detect_objects(self, frame, conf_threshold=0.5):
        """Detect objects in the given frame using YOLOv5."""
        results = self.model(frame)

        if results.xyxy[0].nelement() == 0:  # If no detections, return empty
            return [], [], results  # <- Make sure it returns 3 values

        bbox = results.xyxy[0].cpu().numpy()[:, :4]  # Convert to [x1, y1, x2, y2] format
        class_ids = results.xyxy[0].cpu().numpy()[:, -1].astype(int)  # Class IDs
        confs = results.xyxy[0].cpu().numpy()[:, 4]  # Confidence scores

        # Filter only persons (class 0 in COCO) and apply confidence threshold
        human_bbox = [bbox[i] for i, cid in enumerate(class_ids) if cid == 0 and confs[i] > conf_threshold]
        human_confs = [confs[i] for i, cid in enumerate(class_ids) if cid == 0 and confs[i] > conf_threshold]

        return human_bbox, human_confs, results  # <- Make sure this returns results

