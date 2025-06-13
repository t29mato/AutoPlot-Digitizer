from __future__ import annotations
from typing import List
import numpy as np
import cv2
from .image_loader import ImageLoader
from .plot_detector import PlotDetector
from .axis_label_detector import AxisLabelDetector
from .marker_extractor import MarkerExtractor
from ..domain.models import PlotData, Series, DigitizationResult, AxisInfo
from ..domain.repositories import IPlotExtractor

class OpenCVPlotExtractor(IPlotExtractor):
    def __init__(self, x_range=(0, 100), y_range=(0, 100)):
        self._plot_detector = PlotDetector()
        self._axis_label_detector = AxisLabelDetector()
        self._marker_extractor = MarkerExtractor()
        self._x_range = x_range
        self._y_range = y_range

    def _calibrate_axes(self, plot_img):
        """Convert pixel coordinates to data coordinates"""
        h, w = plot_img.shape[:2]

        # Assume plot area uses most of the image (with some margin)
        margin_x = int(0.1 * w)  # 10% margin on sides
        margin_y = int(0.1 * h)  # 10% margin on top/bottom

        # Define pixel boundaries of the actual plot area
        x_pixel_min = margin_x
        x_pixel_max = w - margin_x
        y_pixel_min = margin_y
        y_pixel_max = h - margin_y

        # Data value ranges
        x_data_min, x_data_max = self._x_range
        y_data_min, y_data_max = self._y_range

        return {
            'x_pixel_range': (x_pixel_min, x_pixel_max),
            'y_pixel_range': (y_pixel_min, y_pixel_max),
            'x_data_range': (x_data_min, x_data_max),
            'y_data_range': (y_data_min, y_data_max)
        }

    def _calibrate_axes_with_area(self, plot_img, plot_area):
        """Convert pixel coordinates to data coordinates using detected plot area"""
        # Use the detected plot area instead of default margins
        x_pixel_min = plot_area['plot_left']
        x_pixel_max = plot_area['plot_right']
        y_pixel_min = plot_area['plot_top']
        y_pixel_max = plot_area['plot_bottom']

        # Data value ranges
        x_data_min, x_data_max = self._x_range
        y_data_min, y_data_max = self._y_range

        return {
            'x_pixel_range': (x_pixel_min, x_pixel_max),
            'y_pixel_range': (y_pixel_min, y_pixel_max),
            'x_data_range': (x_data_min, x_data_max),
            'y_data_range': (y_data_min, y_data_max)
        }

    def _get_default_margins(self, width, height):
        """Get default plot area margins"""
        return {
            'left': int(0.12 * width),    # Space for Y-axis labels
            'right': int(0.05 * width),   # Small margin
            'top': int(0.05 * height),    # Small margin
            'bottom': int(0.12 * height)  # Space for X-axis labels
        }

    def _detect_axis_lines(self, gray_image):
        """Detect axis lines using Hough line detection"""
        try:
            # Use Hough line detection to find axis lines
            edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            return lines
        except Exception:
            # If cv2 is not available or operations fail, return None
            return None

    def _is_vertical_line(self, theta):
        """Check if a line is vertical based on its angle"""
        return abs(theta) < 0.2 or abs(theta - np.pi) < 0.2

    def _is_horizontal_line(self, theta):
        """Check if a line is horizontal based on its angle"""
        return abs(theta - np.pi/2) < 0.2

    def _is_valid_vertical_line(self, x, width):
        """Check if vertical line is within valid range"""
        return x and 50 < x < width - 50

    def _is_valid_horizontal_line(self, y, height):
        """Check if horizontal line is within valid range"""
        return y and 50 < y < height - 50

    def _process_line(self, rho, theta, width, height):
        """Process a single line and classify it as vertical or horizontal"""
        if self._is_vertical_line(theta):
            x = int(rho / np.cos(theta)) if np.cos(theta) != 0 else None
            if self._is_valid_vertical_line(x, width):
                return 'vertical', x
        elif self._is_horizontal_line(theta):
            y = int(rho / np.sin(theta)) if np.sin(theta) != 0 else None
            if self._is_valid_horizontal_line(y, height):
                return 'horizontal', y
        return None, None

    def _classify_lines(self, lines, width, height):
        """Classify detected lines as vertical or horizontal"""
        vertical_lines = []
        horizontal_lines = []

        if lines is None:
            return vertical_lines, horizontal_lines

        for rho, theta in lines[:, 0]:
            line_type, coordinate = self._process_line(rho, theta, width, height)
            if line_type == 'vertical':
                vertical_lines.append(coordinate)
            elif line_type == 'horizontal':
                horizontal_lines.append(coordinate)

        return vertical_lines, horizontal_lines

    def _refine_margins_with_lines(self, default_margins, vertical_lines, horizontal_lines, height):
        """Refine margins based on detected axis lines"""
        left_margin = default_margins['left']
        bottom_margin = default_margins['bottom']

        # Use detected lines to refine margins
        if vertical_lines:
            left_axis = min(vertical_lines)
            left_margin = max(left_margin, left_axis + 10)

        if horizontal_lines:
            bottom_axis = max(horizontal_lines)
            bottom_margin = max(bottom_margin, height - bottom_axis + 10)

        return left_margin, bottom_margin

    def _detect_plot_area(self, plot_img):
        """Detect the actual plot area excluding axis labels and legends"""
        h, w = plot_img.shape[:2]

        # Get default margins
        default_margins = self._get_default_margins(w, h)

        # Try to detect axis lines (gracefully handle cv2 import issues)
        try:
            gray = cv2.cvtColor(plot_img, cv2.COLOR_BGR2GRAY)
            lines = self._detect_axis_lines(gray)
            vertical_lines, horizontal_lines = self._classify_lines(lines, w, h)
            left_margin, bottom_margin = self._refine_margins_with_lines(
                default_margins, vertical_lines, horizontal_lines, h
            )
        except Exception:
            # Fall back to default margins if cv2 operations fail
            left_margin = default_margins['left']
            bottom_margin = default_margins['bottom']

        return {
            'plot_left': left_margin,
            'plot_right': w - default_margins['right'],
            'plot_top': default_margins['top'],
            'plot_bottom': h - bottom_margin,
            'full_width': w,
            'full_height': h
        }

    def _filter_data_points(self, raw_points, plot_area):
        """Filter out points that are likely labels or legends"""
        filtered_points = {}

        plot_left = plot_area['plot_left']
        plot_right = plot_area['plot_right']
        plot_top = plot_area['plot_top']
        plot_bottom = plot_area['plot_bottom']

        for color_name, points in raw_points.items():
            valid_points = []

            for px, py in points:
                # Only keep points within the actual plot area
                if (plot_left <= px <= plot_right and
                    plot_top <= py <= plot_bottom):
                    valid_points.append((px, py))

            if valid_points:
                filtered_points[color_name] = valid_points

        return filtered_points

    def _pixel_to_data(self, pixel_x, pixel_y, calibration):
        """Convert pixel coordinates to data coordinates"""
        x_pixel_min, x_pixel_max = calibration['x_pixel_range']
        y_pixel_min, y_pixel_max = calibration['y_pixel_range']
        x_data_min, x_data_max = calibration['x_data_range']
        y_data_min, y_data_max = calibration['y_data_range']

        # Convert X coordinate
        x_ratio = (pixel_x - x_pixel_min) / (x_pixel_max - x_pixel_min)
        data_x = x_data_min + x_ratio * (x_data_max - x_data_min)

        # Convert Y coordinate (flip because image Y is inverted)
        y_ratio = (y_pixel_max - pixel_y) / (y_pixel_max - y_pixel_min)
        data_y = y_data_min + y_ratio * (y_data_max - y_data_min)

        return data_x, data_y

    def extract(self, image_path: str):
        img = ImageLoader.read(image_path)
        boxes = self._plot_detector.detect_plots(img)
        plots: List[PlotData] = []

        for pid, (x, y, w, h) in enumerate(boxes):
            crop = img[y:y+h, x:x+w]

            # Detect the actual plot area to avoid labels
            plot_area = self._detect_plot_area(crop)

            # Get calibration for this plot (use plot area for better accuracy)
            calibration = self._calibrate_axes_with_area(crop, plot_area)

            # Detect axis labels from strips left and bottom
            left_strip = crop[:, :int(0.15*w)]
            bottom_strip = crop[int(0.85*h):, :]
            y_label, y_unit = self._axis_label_detector.detect(left_strip)
            x_label, x_unit = self._axis_label_detector.detect(bottom_strip)

            # Extract markers in pixel coordinates
            raw_pts = self._marker_extractor.extract(crop)

            # Filter out points that are likely axis labels or legends
            filtered_pts = self._filter_data_points(raw_pts, plot_area)

            # Convert pixel coordinates to data coordinates
            series_objs: List[Series] = []
            for name, pts in filtered_pts.items():
                converted_points = []
                for px, py in pts:
                    data_x, data_y = self._pixel_to_data(px, py, calibration)
                    converted_points.append((float(data_x), float(data_y)))

                series_objs.append(Series(sample=name, points=converted_points))

            plots.append(PlotData(
                id=pid,
                x_axis=AxisInfo(label=x_label, unit=x_unit),
                y_axis=AxisInfo(label=y_label, unit=y_unit),
                series=series_objs
            ))

        return DigitizationResult(plots=plots)
