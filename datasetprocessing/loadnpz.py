import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.abspath("..")) 

from preprocess.preprocess import Preprocess


class LoadNPZ:
    def __init__(self, video_path, output_npz):
        self.video_path = video_path
        self.output_npz = output_npz
        self.preprocessor = Preprocess()

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        skeleton_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = self.preprocessor.preprocess_frame(frame)
            
            # Extract skeleton landmarks
            skeleton_landmarks = self.preprocessor.skeletal_mapper.map_skeletons(frame)
            
            if skeleton_landmarks:
                skeleton = np.array([[lm.x, lm.y] for lm in skeleton_landmarks.landmark])
                skeleton_data.append(skeleton)

        cap.release()

        # Save skeleton data to NPZ file
        np.savez(self.output_npz, data=np.array(skeleton_data))
        print(f"Saved {len(skeleton_data)} frames to {self.output_npz}")

# Example Usage
video_path = "image.png"
output_npz = "skeleton_data.npz"
loader = LoadNPZ(video_path, output_npz)
loader.process_video()
