"""MIDI output via python-rtmidi with virtual port support."""
from __future__ import annotations
import platform
import time
import mido
import mido.backends.rtmidi  # noqa: F401 — registers backend


class MidiOutput:
    """Sends MIDI CC and note messages via a virtual or named port."""

    def __init__(self, port_name: str = "CV-MIDI", rate_limit: int = 60):
        self._port_name = port_name
        self._min_interval = 1.0 / rate_limit
        self._last_sent: dict[tuple[int, int], float] = {}
        self._port: mido.ports.BaseOutput | None = None

    def open(self):
        """Open a virtual MIDI output port, falling back to console mode in WSL2/no-ALSA envs."""
        try:
            self._port = mido.open_output(self._port_name, virtual=True)
            print(f"MIDI port opened: {self._port_name}")
        except Exception:
            try:
                available = mido.get_output_names()
                if available:
                    self._port = mido.open_output(available[0])
                    print(f"MIDI port opened: {available[0]}")
                    return
            except Exception:
                pass
            # No MIDI hardware available (e.g. WSL2) — run in console mode
            print(
                "WARNING: No MIDI output available (ALSA not found — running in WSL2?).\n"
                "Running in console mode: MIDI messages will be printed to stdout.\n"
                "To send real MIDI, run this on Windows/macOS directly or set up a MIDI bridge."
            )
            self._port = None  # send_cc will print instead

    def close(self):
        if self._port:
            self._port.close()

    def send_cc(self, channel: int, cc: int, value: int):
        """Send a MIDI CC message with rate limiting."""
        channel = max(0, min(15, channel))
        cc = max(0, min(127, cc))
        value = max(0, min(127, value))

        key = (channel, cc)
        now = time.monotonic()
        if now - self._last_sent.get(key, 0.0) < self._min_interval:
            return
        self._last_sent[key] = now

        msg = mido.Message("control_change", channel=channel, control=cc, value=value)
        if self._port:
            self._port.send(msg)
        else:
            print(f"[MIDI] CC ch={channel} cc={cc} val={value}")

    def send_note(self, channel: int, note: int, velocity: int, note_on: bool = True):
        """Send a MIDI note on or note off message."""
        channel = max(0, min(15, channel))
        note = max(0, min(127, note))
        velocity = max(0, min(127, velocity))

        msg_type = "note_on" if note_on else "note_off"
        msg = mido.Message(msg_type, channel=channel, note=note, velocity=velocity)
        if self._port:
            self._port.send(msg)
        else:
            print(f"[MIDI] {msg_type} ch={channel} note={note} vel={velocity}")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()
