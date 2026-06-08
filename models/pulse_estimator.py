import numpy as np
from collections import deque

from config import (
    BUFFER_SIZE,
    HR_SMOOTHING_WINDOW
)


class PulseEstimator:

    def __init__(self):

        self.signal_buffer = deque(
            maxlen=BUFFER_SIZE
        )

        self.hr_buffer = deque(
            maxlen=HR_SMOOTHING_WINDOW
        )

    def add_sample(self, value):

        self.signal_buffer.append(
            value
        )

    def get_signal(self):

        return list(
            self.signal_buffer
        )

    def smooth_bpm(self, bpm):

        if bpm is None:
            return None

        self.hr_buffer.append(
            bpm
        )

        return round(
            np.mean(
                self.hr_buffer
            ),
            1
        )

    def reset(self):

        self.signal_buffer.clear()
        self.hr_buffer.clear()