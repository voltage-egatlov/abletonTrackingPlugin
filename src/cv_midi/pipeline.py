"""Detector pipeline — runs enabled detectors and merges features."""
from __future__ import annotations
import time
import numpy as np
from cv_midi.config import DetectorConfig
from cv_midi.detectors.motion import MotionDetector


class Pipeline:
    """Runs configured detectors on a frame and returns merged feature dict."""

    def __init__(self, config: DetectorConfig):
        self._detectors = {}
        if config.motion:
            self._detectors["motion"] = MotionDetector()
        # Pose and hands detectors added in Phase 2
        if config.pose:
            from cv_midi.detectors.pose import PoseDetector
            self._detectors["pose"] = PoseDetector()
        if config.hands:
            from cv_midi.detectors.hands import HandDetector
            self._detectors["hands"] = HandDetector()
        if config.color:
            from cv_midi.detectors.color import ColorDetector
            self._detectors["color"] = ColorDetector()

    def process(self, frame: np.ndarray) -> dict[str, float]:
        """Run all detectors and return merged feature dict."""
        features: dict[str, float] = {}
        for detector in self._detectors.values():
            result = detector.detect(frame)
            features.update(result)
        return features
