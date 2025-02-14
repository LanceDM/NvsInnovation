import cv2
import mediapipe as mp
import preprocess
# Initialize MediaPipe BlazePose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # For drawing keypoints on the image
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)


# If camera is not found: Open webcam
if cap == None:
  cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert BGR image to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with BlazePose
    results = pose.process(rgb_frame)

    # Draw landmarks if detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show output
    cv2.imshow("BlazePose Detection", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
