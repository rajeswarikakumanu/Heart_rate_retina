from flask import Flask
from flask import render_template
from flask import Response

import cv2
import time
import numpy as np

from utils.eye_tracking import EyeTracker
from utils.retina_roi import RetinaROI
from utils.signal_processing import SignalProcessor

from models.blink_detector import BlinkDetector
from models.pulse_estimator import PulseEstimator

from config import *

app = Flask(__name__)

eye_tracker = EyeTracker()
roi_processor = RetinaROI()

signal_processor = SignalProcessor()

blink_detector = BlinkDetector()
pulse_estimator = PulseEstimator()

cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)

start_time = time.time()

last_blink_count = 0


def reset_metrics():

    pulse_estimator.reset()

    return None, None


def generate_frames():

    global last_blink_count

    frequency = None
    bpm = None

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(
            frame,
            1
        )

        eye_status = "Not Detected"

        landmarks = eye_tracker.get_landmarks(
            frame
        )

        if landmarks is not None:

            left_eye, right_eye = \
                eye_tracker.extract_eye_points(
                    frame,
                    landmarks
                )

            visible = eye_tracker.eyes_visible(
                left_eye,
                right_eye
            )

            if visible:

                eye_status = "Detected"
                left_eye_pts = np.array(left_eye, dtype=np.int32)
                right_eye_pts = np.array(right_eye, dtype=np.int32)
                cv2.polylines(
                     frame,
                    [left_eye_pts],
                     True,
                    (0, 255, 0),
                    2
                )

                cv2.polylines(
                     frame,
                    [right_eye_pts],
                     True,
                    (0, 255, 0),
                    2
                )
                lx, ly, lw, lh = cv2.boundingRect(left_eye_pts)
                rx, ry, rw, rh = cv2.boundingRect(right_eye_pts)
                cv2.rectangle(
                    frame,
                    (lx - 5, ly - 5),
                    (lx + lw + 5, ly + lh + 5),
                    (255, 255, 0),
                     2
                 )
                cv2.rectangle(
                    frame,
                    (rx - 5, ry - 5),
                    (rx + rw + 5, ry + rh + 5),
                    (255, 255, 0),
                     2
                 )
                cv2.putText(
                    frame,
                    "RIGHT EYE",
                    (rx, ry - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )
                cv2.rectangle(
                   frame,
                (
                lx + int(lw * 0.25),
                ly + int(lh * 0.25)
      ),
    (
        lx + int(lw * 0.75),
        ly + int(lh * 0.75)
    ),
    (0, 0, 255),
    2
)
                cv2.rectangle(
    frame,
    (
        rx + int(rw * 0.25),
        ry + int(rh * 0.25)
    ),
    (
        rx + int(rw * 0.75),
        ry + int(rh * 0.75)
    ),
    (0, 0, 255),
    2
)
                left_center = (
    int(np.mean(left_eye_pts[:, 0])),
    int(np.mean(left_eye_pts[:, 1]))
)
                right_center = (
    int(np.mean(right_eye_pts[:, 0])),
    int(np.mean(right_eye_pts[:, 1]))
)
                cv2.circle(
    frame,
    left_center,
    4,
    (0, 0, 255),
    -1
)
                cv2.circle(
    frame,
    right_center,
    4,
    (0, 0, 255),
    -1
)
                

                






                left_roi = roi_processor.extract_roi(
                    frame,
                    left_eye
                )

                right_roi = roi_processor.extract_roi(
                    frame,
                    right_eye
                )

                left_roi = roi_processor.stabilize_roi(
                    left_roi
                )

                right_roi = roi_processor.stabilize_roi(
                    right_roi
                )

                left_roi = roi_processor.illumination_normalization(
                    left_roi
                )

                right_roi = roi_processor.illumination_normalization(
                    right_roi
                )

                rgb1 = roi_processor.get_mean_rgb(
                    left_roi
                )

                rgb2 = roi_processor.get_mean_rgb(
                    right_roi
                )

                if rgb1 is not None and rgb2 is not None:

                    mean_signal = (
                        rgb1[1] +
                        rgb2[1]
                    ) / 2

                    pulse_estimator.add_sample(
                        mean_signal
                    )

                ear, blink_count = \
                    blink_detector.update(
                        left_eye,
                        right_eye
                    )

                elapsed = (
                    time.time()
                    - start_time
                ) / 60

                blink_frequency = 0

                if elapsed > 0:

                    blink_frequency = round(
                        blink_count /
                        elapsed,
                        1
                    )

                signal = pulse_estimator.get_signal()

                if len(signal) > 150:

                    filtered = \
                        signal_processor.preprocess_signal(
                            signal,
                            FPS_ESTIMATE
                        )

                    frequency = \
                        signal_processor.estimate_frequency(
                            filtered,
                            FPS_ESTIMATE
                        )

                    bpm = \
                        signal_processor.calculate_bpm(
                            frequency
                        )

                    bpm = pulse_estimator.smooth_bpm(
                        bpm
                    )

            else:

                frequency, bpm = reset_metrics()

        else:

            frequency, bpm = reset_metrics()

        cv2.putText(
            frame,
            f"Eye Status : {eye_status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            GREEN if eye_status ==
            "Detected" else RED,
            2
        )

        if eye_status == "Detected":

            cv2.putText(
                frame,
                f"Blink Count : {blink_detector.blink_count}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                WHITE,
                2
            )

            cv2.putText(
                frame,
                f"Blink Frequency : {blink_frequency:.1f}/min",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                WHITE,
                2
            )

            cv2.putText(
                frame,
                f"Frequency : {frequency:.2f} Hz"
                if frequency
                else "Frequency : N/A",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                WHITE,
                2
            )

            cv2.putText(
                frame,
                f"Heart Rate : {bpm:.1f} BPM"
                if bpm
                else "Heart Rate : N/A",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                WHITE,
                2
            )

        else:

            cv2.putText(
                frame,
                "Blink Frequency : N/A",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                RED,
                2
            )

            cv2.putText(
                frame,
                "Frequency : N/A",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                RED,
                2
            )

            cv2.putText(
                frame,
                "Heart Rate : N/A",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                RED,
                2
            )

        ret, buffer = cv2.imencode(
            '.jpg',
            frame
        )

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame +
            b'\r\n'
        )


@app.route('/')

def index():

    return render_template(
        'index.html'
    )


@app.route('/video_feed')

def video_feed():

    return Response(
        generate_frames(),
        mimetype=
        'multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )