import pyworld as pw
import numpy as np


class Synthesizer:
    @staticmethod
    def pyworld_synthesize(f0, sp, ap, fs, frame_period):
        synthesized = pw.synthesize(f0, sp, ap, fs, frame_period)
        return synthesized.astype(np.float32)
