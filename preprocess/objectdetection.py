import torch
from ultralytics import YOLO

class ObjectDetection:
    def __init__(self, model_name="yolov8n", device="cpu"):   # either yolov8n or yolov5s
        """Load YOLOv5 or YOLOv8 model dynamically."""
        self.device = device
        self.model_name = model_name
        self.model = self.load_model(model_name)

    def load_model(self, model_name):
        """Automatically detect and load YOLOv5 or YOLOv8."""
        if "yolov5" in model_name:  # Load YOLOv5
            return torch.hub.load("ultralytics/yolov5", model_name, pretrained=True, device=self.device)
        else:  
            return YOLO(model_name)  # Example: "yolov8n.pt"

    def set_model(self, new_model_name):
        """Dynamically change the YOLO model."""
        self.model_name = new_model_name
        self.model = self.load_model(new_model_name)
        print(f"🔄 Switched YOLO model to {new_model_name}")

    def detect_objects(self, frame, conf_threshold=0.5):
        """Detect objects in the given frame using YOLOv5 or YOLOv8."""
        if "yolov5" in self.model_name:
            results = self.model(frame)
            if results.xyxy[0].nelement() == 0:  
                return [], [], results

            bbox = results.xyxy[0].cpu().numpy()[:, :4]  
            class_ids = results.xyxy[0].cpu().numpy()[:, -1].astype(int)  
            confs = results.xyxy[0].cpu().numpy()[:, 4]  
        
        else:  # YOLOv8
            results = self.model(frame)
            detections = results[0].boxes.data.cpu().numpy()  # (x1, y1, x2, y2, conf, class)
            bbox = detections[:, :4]
            confs = detections[:, 4]
            class_ids = detections[:, 5].astype(int)

        # Filter only persons (class 0 in COCO)
        human_bbox = [bbox[i] for i, cid in enumerate(class_ids) if cid == 0 and confs[i] > conf_threshold]
        human_confs = [confs[i] for i, cid in enumerate(class_ids) if cid == 0 and confs[i] > conf_threshold]

        return human_bbox, human_confs, results
