"""Tests for the detector pipeline."""
import numpy as np
import pytest
from cv_midi.config import DetectorConfig
from cv_midi.pipeline import Pipeline


def _blank_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _white_frame():
    return np.ones((480, 640, 3), dtype=np.uint8) * 255


class TestPipeline:
    def test_motion_only_returns_motion_intensity(self):
        config = DetectorConfig(motion=True, pose=False, hands=False, color=False)
        pipeline = Pipeline(config)
        result = pipeline.process(_blank_frame())
        assert "motion_intensity" in result

    def test_no_detectors_returns_empty(self):
        config = DetectorConfig(motion=False, pose=False, hands=False, color=False)
        pipeline = Pipeline(config)
        result = pipeline.process(_blank_frame())
        assert result == {}

    def test_motion_zero_on_first_frame(self):
        config = DetectorConfig(motion=True)
        pipeline = Pipeline(config)
        result = pipeline.process(_blank_frame())
        assert result["motion_intensity"] == 0.0

    def test_motion_detected_on_second_frame(self):
        config = DetectorConfig(motion=True)
        pipeline = Pipeline(config)
        pipeline.process(_blank_frame())
        result = pipeline.process(_white_frame())
        assert result["motion_intensity"] > 0.5

    def test_all_values_normalized(self):
        config = DetectorConfig(motion=True)
        pipeline = Pipeline(config)
        pipeline.process(_blank_frame())
        result = pipeline.process(_white_frame())
        for value in result.values():
            assert 0.0 <= value <= 1.0

    def test_color_detector_enabled(self):
        config = DetectorConfig(motion=False, color=True)
        pipeline = Pipeline(config)
        result = pipeline.process(_blank_frame())
        assert "brightness" in result
        assert "saturation" in result
        assert "hue" in result

    def test_color_brightness_black_frame(self):
        config = DetectorConfig(motion=False, color=True)
        pipeline = Pipeline(config)
        result = pipeline.process(_blank_frame())
        assert result["brightness"] == pytest.approx(0.0, abs=0.01)

    def test_color_brightness_white_frame(self):
        config = DetectorConfig(motion=False, color=True)
        pipeline = Pipeline(config)
        result = pipeline.process(_white_frame())
        assert result["brightness"] == pytest.approx(1.0, abs=0.01)

    def test_motion_and_color_together(self):
        config = DetectorConfig(motion=True, color=True)
        pipeline = Pipeline(config)
        result = pipeline.process(_blank_frame())
        assert "motion_intensity" in result
        assert "brightness" in result
