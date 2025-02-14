import cv2
from preprocess import Preprocess

# Initialize preprocess class
preprocess = Preprocess(device="cpu")  # Use the appropriate device (cpu or cuda)

# Read the image you want to process (change the path to your image file)
image_path = "image.png"  
frame = cv2.imread(image_path)

# Preprocess the frame (detect objects and map skeletons)
processed_frame = preprocess.preprocess_frame(frame)


# Display the result (image with bounding boxes and skeletons)
cv2.imshow("Processed Image", processed_frame)

# Wait for a key press and close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()
