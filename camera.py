import cv2

class Camera:
    def __init__(self, camera_index=0):
        """Initialize the camera with the given index (default is 0)."""
        self.camera_index = camera_index
        self.cap = None
    
    def open(self):
        """Open the camera if not already opened."""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print("Error: Unable to access the camera.")
            else:
                print("Camera initialized successfully.")
        else:
            print("Camera already opened.")
    
    def get_frame(self):
        """Capture a frame from the camera."""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                print("Error: Failed to capture frame.")
        else:
            print("Error: Camera is not opened.")
            return None

    def release(self):
        """Release the camera and close it."""
        if self.cap is not None:
            self.cap.release()
            print("Camera released.")
        else:
            print("Camera is not initialized.")
