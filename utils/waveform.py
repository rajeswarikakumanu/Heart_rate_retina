import numpy as np


class Waveform:

    def __init__(self):

        self.values = []

    def update(self, value):

        self.values.append(value)

        if len(self.values) > 300:
            self.values.pop(0)

    def get_wave(self):

        return self.values

    def reset(self):

        self.values = []