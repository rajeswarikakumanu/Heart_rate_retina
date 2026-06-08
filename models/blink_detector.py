import numpy as np

from scipy.spatial import distance

from config import (
    EAR_THRESHOLD,
    EAR_CONSEC_FRAMES
)


class BlinkDetector:

    def __init__(self):

        self.blink_count = 0

        self.frame_counter = 0

    def eye_aspect_ratio(
        self,
        eye
    ):

        A = distance.euclidean(
            eye[1],
            eye[5]
        )

        B = distance.euclidean(
            eye[2],
            eye[4]
        )

        C = distance.euclidean(
            eye[0],
            eye[3]
        )

        ear = (A + B) / (2.0 * C)

        return ear

    def update(
        self,
        left_eye,
        right_eye
    ):

        left_ear = self.eye_aspect_ratio(
            left_eye
        )

        right_ear = self.eye_aspect_ratio(
            right_eye
        )

        ear = (
            left_ear +
            right_ear
        ) / 2.0

        if ear < EAR_THRESHOLD:

            self.frame_counter += 1

        else:

            if self.frame_counter >= EAR_CONSEC_FRAMES:

                self.blink_count += 1

            self.frame_counter = 0

        return (
            ear,
            self.blink_count
        )