"""Tests for MIDI output module."""
import time
import pytest
from unittest.mock import MagicMock, patch, call
from cv_midi.midi_output import MidiOutput


class TestMidiOutput:
    def _make_midi(self, rate_limit=1000):
        midi = MidiOutput(port_name="test-port", rate_limit=rate_limit)
        midi._port = MagicMock()
        return midi

    def test_send_cc_calls_port(self):
        midi = self._make_midi()
        midi.send_cc(0, 1, 64)
        midi._port.send.assert_called_once()
        msg = midi._port.send.call_args[0][0]
        assert msg.type == "control_change"
        assert msg.channel == 0
        assert msg.control == 1
        assert msg.value == 64

    def test_send_cc_clamps_value_above_127(self):
        midi = self._make_midi()
        midi.send_cc(0, 1, 200)
        msg = midi._port.send.call_args[0][0]
        assert msg.value == 127

    def test_send_cc_clamps_value_below_zero(self):
        midi = self._make_midi()
        midi.send_cc(0, 1, -10)
        msg = midi._port.send.call_args[0][0]
        assert msg.value == 0

    def test_send_cc_clamps_channel(self):
        midi = self._make_midi()
        midi.send_cc(20, 1, 64)
        msg = midi._port.send.call_args[0][0]
        assert msg.channel == 15

    def test_send_cc_clamps_cc_number(self):
        midi = self._make_midi()
        midi.send_cc(0, 200, 64)
        msg = midi._port.send.call_args[0][0]
        assert msg.control == 127

    def test_rate_limiting_skips_second_call(self):
        midi = self._make_midi(rate_limit=1)  # 1 msg/sec max
        midi.send_cc(0, 1, 64)
        midi.send_cc(0, 1, 80)  # Should be skipped (within 1 second)
        assert midi._port.send.call_count == 1

    def test_rate_limiting_allows_after_interval(self):
        import time
        midi = self._make_midi(rate_limit=500)  # 2ms interval
        midi.send_cc(0, 1, 64)
        time.sleep(0.003)  # Wait longer than the 2ms interval
        midi.send_cc(0, 1, 80)
        assert midi._port.send.call_count == 2

    def test_different_ccs_not_rate_limited_together(self):
        midi = self._make_midi(rate_limit=1)
        midi.send_cc(0, 1, 64)
        midi.send_cc(0, 2, 80)  # Different CC — not rate limited
        assert midi._port.send.call_count == 2

    def test_send_note_on(self):
        midi = self._make_midi()
        midi.send_note(0, 60, 100, note_on=True)
        msg = midi._port.send.call_args[0][0]
        assert msg.type == "note_on"
        assert msg.note == 60
        assert msg.velocity == 100

    def test_send_note_off(self):
        midi = self._make_midi()
        midi.send_note(0, 60, 0, note_on=False)
        msg = midi._port.send.call_args[0][0]
        assert msg.type == "note_off"

    def test_no_port_send_cc_is_noop(self):
        midi = MidiOutput()
        midi._port = None
        midi.send_cc(0, 1, 64)  # Should not raise

    def test_context_manager_closes_port(self):
        midi = MidiOutput()
        with patch.object(midi, "open"), patch.object(midi, "close") as mock_close:
            with midi:
                pass
        mock_close.assert_called_once()
