"""
Legend detector for identifying legend areas in scientific plots.
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple, Dict
import re

class LegendDetector:
    """Detects legend areas in scientific plots"""

    def __init__(self):
        self.common_legend_keywords = [
            'material', 'sample', 'condition', 'temperature', 'pressure',
            'data', 'series', 'group', 'type', 'method', 'treatment'
        ]

    def detect_legend_boxes(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect potential legend areas using various heuristics
        Returns list of (x, y, width, height) tuples
        """
        legend_boxes = []

        # Method 1: Text clustering approach
        text_boxes = self._detect_text_clusters(image)
        legend_boxes.extend(text_boxes)

        # Method 2: Color sample detection
        color_boxes = self._detect_color_samples(image)
        legend_boxes.extend(color_boxes)

        # Method 3: Rectangular region detection
        shape_boxes = self._detect_legend_shapes(image)
        legend_boxes.extend(shape_boxes)

        # Remove duplicates and filter by size
        legend_boxes = self._filter_and_merge_boxes(legend_boxes, image.shape)

        return legend_boxes

    def _detect_text_clusters(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect areas with clustered text that might be legends"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use MSER (Maximally Stable Extremal Regions) to find text-like regions
        mser = cv2.MSER_create()
        mser.setMinArea(30)
        mser.setMaxArea(14400)
        mser.setDelta(5)

        regions, _ = mser.detectRegions(gray)

        if not regions:
            return []

        # Group nearby regions
        boxes = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
            # Filter by aspect ratio and size (typical for text)
            if 0.1 < h/w < 10 and 100 < w*h < 10000:
                boxes.append((x, y, w, h))

        # Cluster nearby boxes
        clustered_boxes = self._cluster_nearby_boxes(boxes)

        return clustered_boxes

    def _detect_color_samples(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect small colored rectangles or lines that might be legend markers"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Look for high saturation areas (colored elements)
        saturation = hsv[:, :, 1]
        high_sat_mask = saturation > 100

        # Find contours in high saturation areas
        contours, _ = cv2.findContours(
            high_sat_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        color_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Look for small rectangular regions (legend markers)
            if 5 < w < 50 and 2 < h < 20 and 50 < w*h < 1000:
                color_boxes.append((x, y, w, h))

        return color_boxes

    def _detect_legend_shapes(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect rectangular regions that might contain legends"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        shape_boxes = []
        h_img, w_img = image.shape[:2]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Look for rectangular regions in typical legend locations
            # (corners, edges of the plot)
            is_corner = (x < w_img * 0.2 or x > w_img * 0.8 or
                        y < h_img * 0.2 or y > h_img * 0.8)

            # Reasonable size for legend
            reasonable_size = (50 < w < w_img * 0.3 and
                              30 < h < h_img * 0.3 and
                              500 < w*h < w_img * h_img * 0.1)

            if is_corner and reasonable_size:
                shape_boxes.append((x, y, w, h))

        return shape_boxes

    def _cluster_nearby_boxes(self, boxes: List[Tuple[int, int, int, int]],
                             distance_threshold: int = 50) -> List[Tuple[int, int, int, int]]:
        """Cluster nearby boxes into single legend areas"""
        if not boxes:
            return []

        clustered = []
        used = [False] * len(boxes)

        for i, box1 in enumerate(boxes):
            if used[i]:
                continue

            cluster = [box1]
            used[i] = True

            for j, box2 in enumerate(boxes):
                if used[j] or i == j:
                    continue

                # Check if boxes are close enough to be in same legend
                if self._boxes_are_close(box1, box2, distance_threshold):
                    cluster.append(box2)
                    used[j] = True

            # Merge boxes in cluster
            if len(cluster) > 1:
                merged_box = self._merge_boxes(cluster)
                clustered.append(merged_box)
            else:
                clustered.append(box1)

        return clustered

    def _boxes_are_close(self, box1: Tuple[int, int, int, int],
                        box2: Tuple[int, int, int, int], threshold: int) -> bool:
        """Check if two boxes are close enough to be in the same legend"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate centers
        center1_x, center1_y = x1 + w1//2, y1 + h1//2
        center2_x, center2_y = x2 + w2//2, y2 + h2//2

        # Calculate distance between centers
        distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)

        return distance < threshold

    def _merge_boxes(self, boxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        """Merge multiple boxes into a single bounding box"""
        min_x = min(box[0] for box in boxes)
        min_y = min(box[1] for box in boxes)
        max_x = max(box[0] + box[2] for box in boxes)
        max_y = max(box[1] + box[3] for box in boxes)

        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def _filter_and_merge_boxes(self, boxes: List[Tuple[int, int, int, int]],
                               image_shape: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """Filter out invalid boxes and merge overlapping ones"""
        h_img, w_img = image_shape[:2]

        # Filter by size and position
        valid_boxes = []
        for x, y, w, h in boxes:
            # Must be within image bounds
            if x >= 0 and y >= 0 and x + w <= w_img and y + h <= h_img:
                # Must have reasonable size
                if 20 < w < w_img * 0.5 and 15 < h < h_img * 0.5:
                    valid_boxes.append((x, y, w, h))

        # Merge overlapping boxes
        merged_boxes = []
        used = [False] * len(valid_boxes)

        for i, box1 in enumerate(valid_boxes):
            if used[i]:
                continue

            overlapping = [box1]
            used[i] = True

            for j, box2 in enumerate(valid_boxes):
                if used[j] or i == j:
                    continue

                if self._boxes_overlap(box1, box2):
                    overlapping.append(box2)
                    used[j] = True

            if len(overlapping) > 1:
                merged_box = self._merge_boxes(overlapping)
                merged_boxes.append(merged_box)
            else:
                merged_boxes.append(box1)

        return merged_boxes

    def _boxes_overlap(self, box1: Tuple[int, int, int, int],
                      box2: Tuple[int, int, int, int]) -> bool:
        """Check if two boxes overlap"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)
