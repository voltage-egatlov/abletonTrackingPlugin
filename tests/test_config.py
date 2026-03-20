"""Tests for configuration loader."""
import pytest
import yaml
import tempfile
import os
from cv_midi.config import load_config, AppConfig


VALID_CONFIG = """
midi:
  port_name: "CV-MIDI"
  rate_limit: 60

camera:
  index: 0
  width: 640
  height: 480
  fps: 30

detectors:
  motion: true
  pose: false
  hands: false
  color: false

mappings:
  motion_intensity:
    cc: 1
    channel: 0
    min_val: 0
    max_val: 127
    smoothing: 0.3
"""


class TestLoadConfig:
    def _write_config(self, content):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_loads_valid_config(self):
        path = self._write_config(VALID_CONFIG)
        try:
            config = load_config(path)
            assert config.midi.port_name == "CV-MIDI"
            assert config.camera.width == 640
            assert config.detectors.motion is True
            assert config.detectors.pose is False
            assert "motion_intensity" in config.mappings
        finally:
            os.unlink(path)

    def test_mapping_values_parsed(self):
        path = self._write_config(VALID_CONFIG)
        try:
            config = load_config(path)
            m = config.mappings["motion_intensity"]
            assert m.cc == 1
            assert m.channel == 0
            assert m.smoothing == 0.3
        finally:
            os.unlink(path)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_invalid_yaml_raises(self):
        path = self._write_config("invalid: yaml: content: [unclosed")
        try:
            with pytest.raises(Exception):
                load_config(path)
        finally:
            os.unlink(path)

    def test_invalid_cc_value_raises(self):
        bad_config = VALID_CONFIG.replace("cc: 1", "cc: 200")
        path = self._write_config(bad_config)
        try:
            with pytest.raises(Exception):
                load_config(path)
        finally:
            os.unlink(path)
