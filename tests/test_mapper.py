"""Tests for feature-to-MIDI mapper."""
import pytest
from cv_midi.mapper import Mapper, MappingConfig


class TestMappingConfig:
    def test_valid_config(self):
        cfg = MappingConfig(cc=1, channel=0, min_val=0, max_val=127, smoothing=0.3)
        assert cfg.cc == 1
        assert cfg.channel == 0
        assert cfg.smoothing == 0.3

    def test_cc_clamped_to_valid_range(self):
        with pytest.raises(ValueError):
            MappingConfig(cc=128, channel=0)

    def test_channel_clamped_to_valid_range(self):
        with pytest.raises(ValueError):
            MappingConfig(cc=1, channel=16)

    def test_smoothing_out_of_range(self):
        with pytest.raises(ValueError):
            MappingConfig(cc=1, channel=0, smoothing=1.5)


class TestMapper:
    def _make_mapper(self, smoothing=0.0):
        config = {
            "motion": MappingConfig(cc=1, channel=0, min_val=0, max_val=127, smoothing=smoothing),
        }
        return Mapper(config)

    def test_maps_zero_feature_to_min(self):
        mapper = self._make_mapper()
        messages = mapper.process({"motion": 0.0})
        assert len(messages) == 1
        assert messages[0] == (0, 1, 0)  # (channel, cc, value)

    def test_maps_one_feature_to_max(self):
        mapper = self._make_mapper()
        messages = mapper.process({"motion": 1.0})
        assert messages[0] == (0, 1, 127)

    def test_maps_midpoint(self):
        mapper = self._make_mapper()
        messages = mapper.process({"motion": 0.5})
        # 0.5 * 127 = 63.5, rounds to 64
        assert messages[0] == (0, 1, 64)

    def test_clamps_above_one(self):
        mapper = self._make_mapper()
        messages = mapper.process({"motion": 1.5})
        assert messages[0] == (0, 1, 127)

    def test_clamps_below_zero(self):
        mapper = self._make_mapper()
        messages = mapper.process({"motion": -0.5})
        assert messages[0] == (0, 1, 0)

    def test_unknown_feature_ignored(self):
        mapper = self._make_mapper()
        messages = mapper.process({"unknown_feature": 0.5})
        assert len(messages) == 0

    def test_smoothing_blends_values(self):
        # With smoothing=1.0, output = previous (no change)
        # With smoothing=0.0, output = current immediately
        config = {
            "motion": MappingConfig(cc=1, channel=0, smoothing=0.5),
        }
        mapper = Mapper(config)
        # First call: no previous, output is current value
        mapper.process({"motion": 0.0})
        # Second call: smoothed = 0.5 * 1.0 + (1-0.5) * 0.0 = 0.5
        messages = mapper.process({"motion": 1.0})
        assert messages[0][2] == pytest.approx(63, abs=1)

    def test_custom_min_max_range(self):
        config = {
            "motion": MappingConfig(cc=1, channel=0, min_val=64, max_val=127, smoothing=0.0),
        }
        mapper = Mapper(config)
        messages = mapper.process({"motion": 0.0})
        assert messages[0][2] == 64
        messages = mapper.process({"motion": 1.0})
        assert messages[0][2] == 127

    def test_multiple_features(self):
        config = {
            "motion": MappingConfig(cc=1, channel=0, smoothing=0.0),
            "brightness": MappingConfig(cc=2, channel=1, smoothing=0.0),
        }
        mapper = Mapper(config)
        messages = mapper.process({"motion": 0.5, "brightness": 1.0})
        assert len(messages) == 2
        ccs = {m[1]: m[2] for m in messages}
        assert ccs[1] == 64  # 0.5 * 127 rounds to 64
        assert ccs[2] == 127
