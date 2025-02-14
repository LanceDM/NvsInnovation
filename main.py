import cv2
import torch
from camera import Camera  # Import camera module
from preprocess import Preprocess  # Import preprocess module

# Define device: 'cuda' for GPU (if available), 'cpu' for CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize camera
camera = Camera()
camera.open()

# Initialize preprocessing with the device specified
preprocess = Preprocess(device=device)

while True:
    frame = camera.get_frame()
    if frame is not None:
        # Run preprocess (object detection + skeletal mapping) on the specified device
        objects, skeletons = preprocess.preprocess_frame(frame)
        
        # Optionally display results or do further processing
        cv2.imshow("Processed Frame", frame)
    
    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera resources
camera.release()
cv2.destroyAllWindows()
