from __future__ import annotations
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple

class MarkerExtractor:
    """Extract marker pixel centroids per color within a given plot box."""
    _hsv_ranges: Dict[str, Tuple[Tuple[int,int,int], Tuple[int,int,int]]] = {
        "black": ((0,0,0), (180,255,50)),
        "red1": ((0, 150, 120), (10,255,255)),
        "red2": ((170,150,120), (180,255,255)),
        "blue": ((90, 80, 110), (130,255,255)),
        "teal": ((80,80,120), (90,255,255)),
        "magenta": ((140,80,120), (165,255,255)),
    }

    def extract(self, plot_img) -> Dict[str, List[Tuple[int,int]]]:
        hsv = cv2.cvtColor(plot_img, cv2.COLOR_BGR2HSV)
        results: Dict[str, List[Tuple[int,int]]] = {}
        for name, (lo, hi) in self._hsv_ranges.items():
            mask = cv2.inRange(hsv, lo, hi)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
            ys, xs = np.where(mask>0)
            pts = np.column_stack((xs, ys))
            if len(pts) == 0:
                continue
            clusters = DBSCAN(eps=5, min_samples=4).fit(pts)
            for cid in set(clusters.labels_):
                if cid == -1:
                    continue
                cluster_pts = pts[clusters.labels_==cid]
                cx, cy = cluster_pts.mean(axis=0)
                results.setdefault(name, []).append((int(cx), int(cy)))
        return results
