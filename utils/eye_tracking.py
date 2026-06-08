import cv2
import mediapipe as mp
import numpy as np

from config import (
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MAX_NUM_FACES
)

# Left Eye Landmarks
LEFT_EYE = [
    33, 160, 158, 133, 153, 144
]

# Right Eye Landmarks
RIGHT_EYE = [
    362, 385, 387, 263, 373, 380
]

class EyeTracker:

    def __init__(self):

        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=MAX_NUM_FACES,
            refine_landmarks=True,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )

    def get_landmarks(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        return results.multi_face_landmarks[0]

    def extract_eye_points(self, frame, landmarks):

        h, w, _ = frame.shape

        left_points = []
        right_points = []

        for idx in LEFT_EYE:
            lm = landmarks.landmark[idx]
            left_points.append((int(lm.x * w), int(lm.y * h)))

        for idx in RIGHT_EYE:
            lm = landmarks.landmark[idx]
            right_points.append((int(lm.x * w), int(lm.y * h)))

        return np.array(left_points), np.array(right_points)

    def eyes_visible(self, left_eye, right_eye):

        if len(left_eye) < 6:
            return False

        if len(right_eye) < 6:
            return False

        return True