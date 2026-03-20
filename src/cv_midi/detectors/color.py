"""Color/brightness detector — Phase 2."""
import numpy as np
import cv2
from cv_midi.detectors.base import BaseDetector


class ColorDetector(BaseDetector):
    def detect(self, frame: np.ndarray) -> dict[str, float]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        return {
            "brightness":  float(np.mean(v) / 255.0),
            "saturation":  float(np.mean(s) / 255.0),
            "hue":         float(np.mean(h) / 179.0),
        }
