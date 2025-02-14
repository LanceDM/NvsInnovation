import cv2
import torch
import ultralytics

# Check if CUDA (GPU) is available and set device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLOv5 model (pre-trained) on the selected device
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device=device)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Perform inference with YOLOv5 on the frame
    results = model(frame)

    # Filter results to show only humans (class 0 corresponds to 'person')
    for *xyxy, conf, cls in results.xyxy[0]:
        if int(cls) == 0:  # Class 0 is 'person'
            # Draw bounding box for detected human
            x1, y1, x2, y2 = map(int, xyxy)
            label = f"Human: {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Webcam Human Detection', frame)

    # Exit on pressing ESC key
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
