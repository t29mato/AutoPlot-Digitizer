from abc import ABC, abstractmethod
from typing import Dict

import numpy as np


class ColorMasker(ABC):
    """Port that converts an image to N binary masks keyed by color name."""

    @abstractmethod
    def mask(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        ...
