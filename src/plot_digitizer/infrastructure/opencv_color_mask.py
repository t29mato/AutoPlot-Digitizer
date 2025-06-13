"""Simple HSV threshold‑based color masker."""

from typing import Dict, Tuple

import cv2
import numpy as np

from plot_digitizer.interfaces.color_mask import ColorMasker


class HSVColorMasker(ColorMasker):
    """Produce binary masks given hard‑coded HSV windows per color name."""

    def __init__(self, hsv_windows: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]]):
        self._win = hsv_windows

    def mask(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masks = {}
        for name, (lo, hi) in self._win.items():
            mask = cv2.inRange(hsv, lo, hi)
            mask = cv2.medianBlur(mask, 3)
            masks[name] = mask
        return masks
