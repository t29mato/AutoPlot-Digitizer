"""Locate individual plots (sub‑axes) in an image.
Currently supports rectangular grids; future: heuristic split.
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple

class PlotDetector:
    def detect_plots(self, image) -> List[Tuple[int,int,int,int]]:
        """Return list of bounding boxes (x, y, w, h)."""
        # strategy: grayscale -> adaptive thresh -> find contours -> filter by aspect ratio
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY_INV, 31, 15)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[Tuple[int,int,int,int]] = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w*h < 5000:  # noise
                continue
            boxes.append((x, y, w, h))
        # sort left‑top to right‑bottom
        boxes.sort(key=lambda b: (b[1], b[0]))
        return boxes
