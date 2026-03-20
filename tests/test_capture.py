"""Tests for VideoCapture — uses mocks to avoid needing a real webcam."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from cv_midi.capture import VideoCapture


class TestVideoCapture:
    def _make_capture(self):
        cap = VideoCapture(index=0, width=640, height=480, fps=30)
        return cap

    def test_get_latest_frame_returns_none_before_start(self):
        cap = self._make_capture()
        assert cap.get_latest_frame() is None

    def test_start_raises_if_camera_not_opened(self):
        cap = self._make_capture()
        mock_cv2_cap = MagicMock()
        mock_cv2_cap.isOpened.return_value = False
        with patch("cv2.VideoCapture", return_value=mock_cv2_cap):
            with pytest.raises(RuntimeError, match="Cannot open camera"):
                cap.start()

    def test_get_latest_frame_returns_copy(self):
        cap = self._make_capture()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cap._frame = frame
        result = cap.get_latest_frame()
        assert result is not frame  # Must be a copy
        assert np.array_equal(result, frame)

    def test_stop_releases_capture(self):
        cap = self._make_capture()
        mock_cv2_cap = MagicMock()
        mock_cv2_cap.isOpened.return_value = True
        mock_cv2_cap.read.return_value = (False, None)
        with patch("cv2.VideoCapture", return_value=mock_cv2_cap):
            cap.start()
            cap.stop()
        mock_cv2_cap.release.assert_called_once()

    def test_context_manager_starts_and_stops(self):
        cap = self._make_capture()
        mock_cv2_cap = MagicMock()
        mock_cv2_cap.isOpened.return_value = True
        mock_cv2_cap.read.return_value = (False, None)
        with patch("cv2.VideoCapture", return_value=mock_cv2_cap):
            with cap:
                pass
        mock_cv2_cap.release.assert_called_once()

    def test_frame_updated_by_background_thread(self):
        import time
        cap = self._make_capture()
        fake_frame = np.ones((480, 640, 3), dtype=np.uint8) * 42
        mock_cv2_cap = MagicMock()
        mock_cv2_cap.isOpened.return_value = True
        mock_cv2_cap.read.return_value = (True, fake_frame)
        with patch("cv2.VideoCapture", return_value=mock_cv2_cap):
            cap.start()
            time.sleep(0.05)  # Let background thread run
            result = cap.get_latest_frame()
            cap.stop()
        assert result is not None
        assert np.array_equal(result, fake_frame)
