"""Tests for motion detector."""
import numpy as np
import pytest
from cv_midi.detectors.motion import MotionDetector


class TestMotionDetector:
    def _blank_frame(self, h=480, w=640):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def _white_frame(self, h=480, w=640):
        return np.ones((h, w, 3), dtype=np.uint8) * 255

    def test_no_motion_between_identical_frames(self):
        detector = MotionDetector()
        frame = self._blank_frame()
        detector.detect(frame)  # seed previous frame
        result = detector.detect(frame)
        assert result["motion_intensity"] == pytest.approx(0.0, abs=0.01)

    def test_full_motion_between_opposite_frames(self):
        detector = MotionDetector()
        detector.detect(self._blank_frame())
        result = detector.detect(self._white_frame())
        assert result["motion_intensity"] > 0.9

    def test_returns_zero_on_first_frame(self):
        detector = MotionDetector()
        result = detector.detect(self._blank_frame())
        assert result["motion_intensity"] == 0.0

    def test_output_normalized_between_zero_and_one(self):
        detector = MotionDetector()
        detector.detect(self._blank_frame())
        result = detector.detect(self._white_frame())
        assert 0.0 <= result["motion_intensity"] <= 1.0

    def test_partial_motion(self):
        detector = MotionDetector()
        blank = self._blank_frame()
        partial = self._blank_frame()
        # Change top-left quarter
        partial[:240, :320] = 255
        detector.detect(blank)
        result = detector.detect(partial)
        # Motion should be roughly 25% (top-left quarter changed)
        assert 0.1 < result["motion_intensity"] < 0.5

    def test_reset_clears_previous_frame(self):
        detector = MotionDetector()
        detector.detect(self._white_frame())
        detector.reset()
        result = detector.detect(self._blank_frame())
        assert result["motion_intensity"] == 0.0
