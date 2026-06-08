import numpy as np

from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.signal import find_peaks

from config import (
    LOWCUT,
    HIGHCUT,
    FILTER_ORDER,
    MIN_BPM,
    MAX_BPM
)


class SignalProcessor:

    def __init__(self):
        pass

    def butter_bandpass_filter(
        self,
        signal,
        lowcut,
        highcut,
        fs,
        order=4
    ):

        nyquist = 0.5 * fs

        low = lowcut / nyquist
        high = highcut / nyquist

        b, a = butter(
            order,
            [low, high],
            btype='band'
        )

        return filtfilt(
            b,
            a,
            signal
        )

    def preprocess_signal(
        self,
        signal,
        fps
    ):

        if len(signal) < fps * 3:
            return None

        signal = np.array(signal)

        signal = signal - np.mean(signal)

        filtered = self.butter_bandpass_filter(
            signal,
            LOWCUT,
            HIGHCUT,
            fps,
            FILTER_ORDER
        )

        return filtered

    def estimate_frequency(
        self,
        signal,
        fps
    ):

        if signal is None:
            return None

        n = len(signal)

        fft_values = np.abs(
            np.fft.rfft(signal)
        )

        frequencies = np.fft.rfftfreq(
            n,
            d=1/fps
        )

        valid = np.where(
            (frequencies >= LOWCUT) &
            (frequencies <= HIGHCUT)
        )

        if len(valid[0]) == 0:
            return None

        peak_idx = valid[0][
            np.argmax(
                fft_values[valid]
            )
        ]

        dominant_frequency = frequencies[
            peak_idx
        ]

        return dominant_frequency

    def calculate_bpm(
        self,
        frequency
    ):

        if frequency is None:
            return None

        bpm = frequency * 60

        if bpm < MIN_BPM:
            return None

        if bpm > MAX_BPM:
            return None

        return round(bpm, 1)