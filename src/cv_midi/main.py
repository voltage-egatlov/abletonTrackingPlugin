"""Main entry point — wires capture, pipeline, mapper, and MIDI output."""
from __future__ import annotations
import math
import time
import sys
import numpy as np
import cv2
from cv_midi.capture import VideoCapture
from cv_midi.config import load_config
from cv_midi.mapper import Mapper
from cv_midi.midi_output import MidiOutput
from cv_midi.pipeline import Pipeline

DEFAULT_CONFIG = "config.yaml"


def main(config_path: str = DEFAULT_CONFIG, demo: bool = False):
    config = load_config(config_path)
    pipeline = Pipeline(config.detectors)
    mapper = Mapper(config.mappings)
    midi = MidiOutput(port_name=config.midi.port_name, rate_limit=config.midi.rate_limit)

    midi.open()

    if demo:
        print("Running in DEMO mode — synthetic frames, no webcam needed. Ctrl+C to stop.")
        _run_demo(pipeline, mapper, midi)
    else:
        capture = VideoCapture(
            index=config.camera.index,
            width=config.camera.width,
            height=config.camera.height,
            fps=config.camera.fps,
        )
        print("Starting capture... Press Q in the preview window or Ctrl+C to stop.")
        capture.start()
        try:
            _run_live(capture, pipeline, mapper, midi)
        finally:
            capture.stop()

    midi.close()
    cv2.destroyAllWindows()
    print("Stopped.")


def _run_live(capture: VideoCapture, pipeline: Pipeline, mapper: Mapper, midi: MidiOutput):
    frame_count = 0
    fps_timer = time.monotonic()
    try:
        while True:
            frame = capture.get_latest_frame()
            if frame is None:
                time.sleep(0.001)
                continue

            t0 = time.monotonic()
            features = pipeline.process(frame)
            messages = mapper.process(features)
            for channel, cc, value in messages:
                midi.send_cc(channel, cc, value)
            latency_ms = (time.monotonic() - t0) * 1000

            frame_count += 1
            elapsed = time.monotonic() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                print(f"FPS: {fps:.1f}  |  Latency: {latency_ms:.1f}ms  |  Features: {features}")
                frame_count = 0
                fps_timer = time.monotonic()

            _draw_overlay(frame, features, latency_ms)
            cv2.imshow("CV-MIDI Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass


def _run_demo(pipeline: Pipeline, mapper: Mapper, midi: MidiOutput):
    """Synthetic frames: brightness pulses like a sine wave to verify MIDI output."""
    frame_count = 0
    fps_timer = time.monotonic()
    t_start = time.monotonic()
    try:
        while True:
            t = time.monotonic() - t_start
            # Slowly oscillating brightness so motion detector fires
            brightness = int((math.sin(t * 0.5) * 0.5 + 0.5) * 255)
            frame = np.full((480, 640, 3), brightness, dtype=np.uint8)

            t0 = time.monotonic()
            features = pipeline.process(frame)
            messages = mapper.process(features)
            for channel, cc, value in messages:
                midi.send_cc(channel, cc, value)
            latency_ms = (time.monotonic() - t0) * 1000

            frame_count += 1
            elapsed = time.monotonic() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                print(f"FPS: {fps:.1f}  |  Latency: {latency_ms:.2f}ms  |  Features: {features}")
                frame_count = 0
                fps_timer = time.monotonic()

            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass


def _draw_overlay(frame, features: dict, latency_ms: float):
    y = 30
    for name, value in features.items():
        bar_w = int(value * 200)
        cv2.rectangle(frame, (10, y - 15), (10 + bar_w, y), (0, 200, 0), -1)
        cv2.putText(frame, f"{name}: {value:.2f}", (220, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
    cv2.putText(frame, f"Latency: {latency_ms:.1f}ms", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)


if __name__ == "__main__":
    args = sys.argv[1:]
    demo_mode = "--demo" in args
    cfg_args = [a for a in args if not a.startswith("--")]
    cfg = cfg_args[0] if cfg_args else DEFAULT_CONFIG
    main(cfg, demo=demo_mode)
