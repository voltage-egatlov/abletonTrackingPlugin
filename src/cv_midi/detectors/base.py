"""Abstract base class for all detectors."""
from abc import ABC, abstractmethod
import numpy as np


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> dict[str, float]:
        """Process a frame and return named features (0.0-1.0)."""
        ...
