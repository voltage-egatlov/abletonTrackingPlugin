"""Threaded video capture — always delivers the latest frame, drops stale ones."""
from __future__ import annotations
import threading
import time
import cv2
import numpy as np


class VideoCapture:
    """Non-blocking webcam capture on a background thread."""

    def __init__(self, index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self._index = index
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        self._cap = cv2.VideoCapture(self._index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self._index}")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()

    def get_latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.001)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
