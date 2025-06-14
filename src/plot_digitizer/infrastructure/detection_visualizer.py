"""
Detection result visualizer for graph digitization process.
Provides visual feedback on what elements were detected during processing.
"""
from __future__ import annotations
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import colorsys

class DetectionVisualizer:
    """Visualizes the detection results of graph digitization process"""

    def __init__(self):
        self.colors = self._generate_distinct_colors()

    def _generate_distinct_colors(self, num_colors=10):
        """Generate visually distinct colors for different series"""
        colors = []
        for i in range(num_colors):
            hue = i / num_colors
            # Use high saturation and value for vibrant colors
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            # Convert to BGR for OpenCV
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        return colors

    def visualize_plot_area(self, image: np.ndarray, plot_area: Dict[str, int]) -> np.ndarray:
        """Draw the detected plot area boundaries"""
        viz_img = image.copy()

        # Draw plot area rectangle
        cv2.rectangle(viz_img,
                     (plot_area['plot_left'], plot_area['plot_top']),
                     (plot_area['plot_right'], plot_area['plot_bottom']),
                     (0, 255, 0), 2)  # Green rectangle

        # Add label
        cv2.putText(viz_img, "Plot Area",
                   (plot_area['plot_left'], plot_area['plot_top'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return viz_img

    def visualize_axis_lines(self, image: np.ndarray, vertical_lines: List[int],
                           horizontal_lines: List[int]) -> np.ndarray:
        """Draw detected axis lines"""
        viz_img = image.copy()
        h, w = image.shape[:2]

        # Draw vertical lines (typically Y-axis)
        for x in vertical_lines:
            cv2.line(viz_img, (x, 0), (x, h), (255, 0, 0), 2)  # Blue lines
            cv2.putText(viz_img, "Y-axis", (x + 5, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Draw horizontal lines (typically X-axis)
        for y in horizontal_lines:
            cv2.line(viz_img, (0, y), (w, y), (0, 0, 255), 2)  # Red lines
            cv2.putText(viz_img, "X-axis", (10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        return viz_img

    def visualize_data_points(self, image: np.ndarray,
                            data_points: Dict[str, List[Tuple[int, int]]]) -> np.ndarray:
        """Draw detected data points with different colors for each series"""
        viz_img = image.copy()

        color_idx = 0
        for series_name, points in data_points.items():
            color = self.colors[color_idx % len(self.colors)]
            color_idx += 1

            # Draw points
            for px, py in points:
                cv2.circle(viz_img, (int(px), int(py)), 4, color, -1)
                cv2.circle(viz_img, (int(px), int(py)), 6, (255, 255, 255), 1)  # White border

            # Add legend for this series
            legend_y = 30 + color_idx * 25
            cv2.circle(viz_img, (20, legend_y), 6, color, -1)
            cv2.putText(viz_img, series_name, (35, legend_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return viz_img

    def visualize_legend_area(self, image: np.ndarray, legend_boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Draw detected legend areas"""
        viz_img = image.copy()

        for i, (x, y, w, h) in enumerate(legend_boxes):
            # Draw legend box
            cv2.rectangle(viz_img, (x, y), (x + w, y + h), (255, 255, 0), 2)  # Cyan rectangle
            cv2.putText(viz_img, f"Legend {i+1}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return viz_img

    def create_detection_overview(self, image: np.ndarray, detection_results: Dict[str, Any]) -> np.ndarray:
        """Create a comprehensive visualization of all detection results"""
        viz_img = image.copy()

        # Draw plot area if available
        if 'plot_area' in detection_results:
            viz_img = self.visualize_plot_area(viz_img, detection_results['plot_area'])

        # Draw axis lines if available
        if 'vertical_lines' in detection_results and 'horizontal_lines' in detection_results:
            viz_img = self.visualize_axis_lines(viz_img,
                                              detection_results['vertical_lines'],
                                              detection_results['horizontal_lines'])

        # Draw data points if available
        if 'data_points' in detection_results:
            viz_img = self.visualize_data_points(viz_img, detection_results['data_points'])

        # Draw legend areas if available
        if 'legend_boxes' in detection_results:
            viz_img = self.visualize_legend_area(viz_img, detection_results['legend_boxes'])

        # Add title
        cv2.putText(viz_img, "Detection Results Overview", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(viz_img, "Detection Results Overview", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

        return viz_img

    def create_step_by_step_visualization(self, image: np.ndarray,
                                        detection_results: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Create step-by-step visualization for each detection phase"""
        visualizations = {}

        # Step 1: Original image
        visualizations['1_original'] = image.copy()

        # Step 2: Plot area detection
        if 'plot_area' in detection_results:
            visualizations['2_plot_area'] = self.visualize_plot_area(image, detection_results['plot_area'])

        # Step 3: Axis detection
        if 'vertical_lines' in detection_results and 'horizontal_lines' in detection_results:
            step3_img = visualizations.get('2_plot_area', image).copy()
            visualizations['3_axis_detection'] = self.visualize_axis_lines(
                step3_img,
                detection_results['vertical_lines'],
                detection_results['horizontal_lines']
            )

        # Step 4: Data points
        if 'data_points' in detection_results:
            step4_img = visualizations.get('3_axis_detection', visualizations.get('2_plot_area', image)).copy()
            visualizations['4_data_points'] = self.visualize_data_points(
                step4_img,
                detection_results['data_points']
            )

        # Step 5: Complete overview
        visualizations['5_complete'] = self.create_detection_overview(image, detection_results)

        return visualizations
