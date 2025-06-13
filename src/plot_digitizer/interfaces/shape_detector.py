from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class ShapeDetector(ABC):
    """Port that converts mask → list[(px,py)] representing centroids per data point."""

    @abstractmethod
    def detect(self, mask: np.ndarray) -> List[Tuple[float, float]]:
        ...
