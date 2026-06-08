import cv2
import numpy as np

from config import ROI_PADDING


class RetinaROI:

    def __init__(self):
        pass

    def extract_roi(self, frame, eye_points):

        x, y, w, h = cv2.boundingRect(
            np.array(eye_points)
        )

        x = max(0, x - ROI_PADDING)
        y = max(0, y - ROI_PADDING)

        roi = frame[
            y:y+h+(ROI_PADDING*2),
            x:x+w+(ROI_PADDING*2)
        ]

        return roi

    def stabilize_roi(self, roi):

        if roi is None:
            return None

        roi = cv2.GaussianBlur(
            roi,
            (5, 5),
            0
        )

        return roi

    def illumination_normalization(self, roi):

        if roi is None:
            return None

        lab = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2LAB
        )

        l, a, b = cv2.split(lab)

        l = cv2.equalizeHist(l)

        merged = cv2.merge((l, a, b))

        normalized = cv2.cvtColor(
            merged,
            cv2.COLOR_LAB2BGR
        )

        return normalized

    def get_mean_rgb(self, roi):

        if roi is None or roi.size == 0:
            return None

        mean_b = np.mean(roi[:, :, 0])
        mean_g = np.mean(roi[:, :, 1])
        mean_r = np.mean(roi[:, :, 2])

        return np.array([
            mean_r,
            mean_g,
            mean_b
        ])