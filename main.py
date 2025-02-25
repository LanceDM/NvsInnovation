import cv2
import torch
from camera import Camera  
from preprocess import Preprocess


# Initialize camera
camera = Camera()
camera.open()

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

# Initialize preprocess
preprocess = Preprocess(device=device)

while True:
    # Get a frame from the camera
    frame = camera.get_frame()
    
    if frame is not None:
        # Process the frame 
        processed_frame = preprocess.preprocess_frame(frame)
        cv2.imshow("Processed Image", processed_frame)
    
    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera resources
camera.release()
cv2.destroyAllWindows()
