"""Frame-differencing motion detector."""
import numpy as np


class MotionDetector:
    """Detects motion intensity between consecutive frames."""

    def __init__(self, threshold: int = 25):
        self._prev_gray: np.ndarray | None = None
        self._threshold = threshold

    def detect(self, frame: np.ndarray) -> dict[str, float]:
        """Return motion_intensity (0.0-1.0). Returns 0.0 on first call."""
        gray = self._to_gray(frame)

        if self._prev_gray is None:
            self._prev_gray = gray
            return {"motion_intensity": 0.0}

        diff = np.abs(gray.astype(np.int16) - self._prev_gray.astype(np.int16))
        changed = np.sum(diff > self._threshold)
        intensity = changed / gray.size
        self._prev_gray = gray

        return {"motion_intensity": float(np.clip(intensity, 0.0, 1.0))}

    def reset(self):
        self._prev_gray = None

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            # Cheap luminance conversion without cv2 import dependency in tests
            return (
                0.299 * frame[:, :, 2] +
                0.587 * frame[:, :, 1] +
                0.114 * frame[:, :, 0]
            ).astype(np.uint8)
        return frame
