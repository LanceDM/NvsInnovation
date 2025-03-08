from preprocess.preprocess import Preprocess
import cv2

preprocessor = Preprocess(device="cpu")
cap = cv2.VideoCapture(0)  # Use webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame, skeletons, ids = preprocessor.preprocess_frame(frame)

    # Display the frame
    cv2.imshow("Live Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
