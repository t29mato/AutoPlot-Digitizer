from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from plot_digitizer.domain.value_objects import LinearScale


class AxisCalibrator(ABC):
    """Port for pixel↔value calibration."""

    @abstractmethod
    def calibrate(self, image: np.ndarray) -> Tuple[LinearScale, LinearScale]:
        """Return X‑scale and Y‑scale."""
