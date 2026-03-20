"""MediaPipe pose detector — Phase 2."""
import numpy as np
from cv_midi.detectors.base import BaseDetector


class PoseDetector(BaseDetector):
    def __init__(self):
        try:
            import mediapipe as mp
            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                enable_segmentation=False,
                min_detection_confidence=0.5,
            )
        except ImportError:
            raise ImportError("mediapipe is required for pose detection: pip install mediapipe")

    def detect(self, frame: np.ndarray) -> dict[str, float]:
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)
        if not results.pose_landmarks:
            return {}

        lm = results.pose_landmarks.landmark
        h, w = frame.shape[:2]

        def x(idx): return lm[idx].x
        def y(idx): return lm[idx].y

        mp_pose = self._mp_pose.PoseLandmark
        return {
            "left_wrist_x":    float(np.clip(x(mp_pose.LEFT_WRIST), 0, 1)),
            "left_wrist_y":    float(np.clip(1.0 - y(mp_pose.LEFT_WRIST), 0, 1)),
            "right_wrist_x":   float(np.clip(x(mp_pose.RIGHT_WRIST), 0, 1)),
            "right_wrist_y":   float(np.clip(1.0 - y(mp_pose.RIGHT_WRIST), 0, 1)),
            "left_elbow_y":    float(np.clip(1.0 - y(mp_pose.LEFT_ELBOW), 0, 1)),
            "right_elbow_y":   float(np.clip(1.0 - y(mp_pose.RIGHT_ELBOW), 0, 1)),
            "head_y":          float(np.clip(1.0 - y(mp_pose.NOSE), 0, 1)),
        }
