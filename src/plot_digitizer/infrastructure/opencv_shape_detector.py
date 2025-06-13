"""Detect filled glyphs via contour clustering + DBSCAN."""

from typing import List, Tuple

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from plot_digitizer import register_shape_detector
from plot_digitizer.interfaces.shape_detector import ShapeDetector


class FilledMarkerDetector(ShapeDetector):
    """Generic detector for solid‑filled scatter markers (■ ● ▲ ▼ ◀)."""

    def __init__(self, eps: float = 4.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples

    def detect(self, mask: np.ndarray) -> List[Tuple[float, float]]:
        # find all white pixels (mask>0) coordinates
        ys, xs = np.where(mask > 0)
        pts = np.vstack([xs, ys]).T
        if len(pts) == 0:
            return []
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(pts)
        centroids = []
        for cid in set(clustering.labels_):
            if cid == -1:
                continue  # noise
            cp = pts[clustering.labels_ == cid]
            centroids.append(cp.mean(axis=0))
        return centroids

# register so CLI can pick it
register_shape_detector("default", FilledMarkerDetector)
