"""MediaPipe hand gesture detector — Phase 2."""
import numpy as np
from cv_midi.detectors.base import BaseDetector


class HandDetector(BaseDetector):
    def __init__(self):
        try:
            import mediapipe as mp
            self._mp_hands = mp.solutions.hands
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except ImportError:
            raise ImportError("mediapipe is required for hand detection: pip install mediapipe")

    def detect(self, frame: np.ndarray) -> dict[str, float]:
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            return {}

        lm = results.multi_hand_landmarks[0].landmark
        finger_spread = self._finger_spread(lm)
        gesture = self._classify_gesture(lm)
        return {
            "hand_x":          float(np.clip(lm[9].x, 0, 1)),
            "hand_y":          float(np.clip(1.0 - lm[9].y, 0, 1)),
            "finger_spread":   finger_spread,
            "is_fist":         1.0 if gesture == "fist" else 0.0,
            "is_open":         1.0 if gesture == "open" else 0.0,
            "is_pinch":        1.0 if gesture == "pinch" else 0.0,
        }

    def _finger_spread(self, lm) -> float:
        tips = [4, 8, 12, 16, 20]
        wrist = np.array([lm[0].x, lm[0].y])
        distances = [np.linalg.norm(np.array([lm[i].x, lm[i].y]) - wrist) for i in tips]
        avg = np.mean(distances)
        return float(np.clip(avg / 0.4, 0, 1))

    def _classify_gesture(self, lm) -> str:
        thumb_tip = np.array([lm[4].x, lm[4].y])
        index_tip = np.array([lm[8].x, lm[8].y])
        pinch_dist = float(np.linalg.norm(thumb_tip - index_tip))
        if pinch_dist < 0.05:
            return "pinch"

        # Check if fingers are curled (tip y > pip y in image coords = curled down)
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        curled = sum(lm[t].y > lm[p].y for t, p in zip(finger_tips, finger_pips))
        if curled >= 3:
            return "fist"
        return "open"
