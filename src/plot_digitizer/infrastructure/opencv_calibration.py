"""Concrete adapter: axis detection via Canny + Hough."""

from typing import Tuple

import cv2
import numpy as np

from plot_digitizer.domain.value_objects import LinearScale
from plot_digitizer.interfaces.calibration import AxisCalibrator


class OpenCVAxisCalibrator(AxisCalibrator):
    def __init__(self, x_value_range=(0.0, 500.0), y_value_range=(0.0, 6.0)):
        self._x_val = x_value_range
        self._y_val = y_value_range

    def calibrate(self, image: np.ndarray) -> Tuple[LinearScale, LinearScale]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=200, maxLineGap=10)
        if lines is None:
            raise RuntimeError("No axis lines detected; adjust thresholds or check image quality.")
        # separate horizontal vs vertical by orientation
        horiz, vert = [], []
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(y1 - y2) < 5:  # horizontal
                horiz.append((x1, y1, x2, y2))
            elif abs(x1 - x2) < 5:  # vertical
                vert.append((x1, y1, x2, y2))
        # choose extreme lines (origin & far edge)
        x0_line = min(vert, key=lambda l: l[0])  # leftmost
        x1_line = max(vert, key=lambda l: l[0])  # rightmost
        y0_line = max(horiz, key=lambda l: l[1])  # bottom
        y1_line = min(horiz, key=lambda l: l[1])  # top
        x_scale = LinearScale(pixel_start=x0_line[0], pixel_end=x1_line[0],
                              value_start=self._x_val[0], value_end=self._x_val[1])
        y_scale = LinearScale(pixel_start=y0_line[1], pixel_end=y1_line[1],
                              value_start=self._y_val[0], value_end=self._y_val[1])
        return x_scale, y_scale
