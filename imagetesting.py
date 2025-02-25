import cv2
from preprocess.preprocess import Preprocess

# Initialize preprocess class
preprocess = Preprocess(device="cpu") 

# Read the image you want to process (change the path to your image file)
image_path = "image.png"  
frame = cv2.imread(image_path)

# Preprocess the frame 
processed_frame = preprocess.preprocess_frame(frame)

cv2.imshow("Processed Image", processed_frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
