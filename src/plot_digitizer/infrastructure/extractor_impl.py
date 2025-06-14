from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import cv2
from .image_loader import ImageLoader
from .plot_detector import PlotDetector
from .axis_label_detector import AxisLabelDetector
from .marker_extractor import MarkerExtractor
from .detection_visualizer import DetectionVisualizer
from .legend_detector import LegendDetector
from ..domain.models import PlotData, Series, DigitizationResult, AxisInfo
from ..domain.repositories import IPlotExtractor

class OpenCVPlotExtractor(IPlotExtractor):
    def __init__(self, x_range=(0, 100), y_range=(0, 100)):
        self._plot_detector = PlotDetector()
        self._axis_label_detector = AxisLabelDetector()
        self._marker_extractor = MarkerExtractor()
        self._visualizer = DetectionVisualizer()
        self._legend_detector = LegendDetector()
        self._x_range = x_range
        self._y_range = y_range
        # Store intermediate results for visualization
        self._detection_results = {}

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
            'left': int(0.12 * width),     # Space for Y-axis labels
            'right': int(0.25 * width),    # Larger margin for legends on the right
            'top': int(0.08 * height),     # Small margin for titles
            'bottom': int(0.15 * height)   # Space for X-axis labels
        }

    def _detect_axis_lines(self, gray_image):
        """Detect axis lines using Hough line detection"""
        try:
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)

            # Use more sensitive edge detection for axis lines
            edges = cv2.Canny(blurred, 30, 100, apertureSize=3)

            # Use more permissive parameters for Hough line detection
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
            return lines
        except Exception:
            # If cv2 is not available or operations fail, return None
            return None

    def _is_vertical_line(self, theta):
        """Check if a line is vertical based on its angle"""
        return abs(theta) < 0.3 or abs(theta - np.pi) < 0.3

    def _is_horizontal_line(self, theta):
        """Check if a line is horizontal based on its angle"""
        return abs(theta - np.pi/2) < 0.3

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

    def _detect_legend_area_bounds(self, plot_img):
        """Detect legend area on the right side to better define plot bounds"""
        _, w = plot_img.shape[:2]

        # Focus on the right portion of the image where legends typically appear
        right_portion = plot_img[:, int(0.7 * w):]

        # Convert to grayscale for analysis
        gray_right = cv2.cvtColor(right_portion, cv2.COLOR_BGR2GRAY)

        # Look for text-like regions (legends)
        # Use contour detection to find rectangular regions
        _, binary = cv2.threshold(gray_right, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        legend_left_bound = w  # Default to full width if no legend found

        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            # Adjust x coordinate to full image coordinate system
            x_full = x + int(0.7 * w)

            # Look for rectangular regions that could be legends
            # (decent size, reasonable aspect ratio)
            if cw > 20 and ch > 15 and cw * ch > 300:
                # Check if this region contains varied colors (typical of legends)
                roi = plot_img[y:y+ch, x_full:x_full+cw]
                if roi.size > 0:
                    # Simple heuristic: if the region has color variation, it might be a legend
                    color_variance = np.std(roi)
                    if color_variance > 20:  # Threshold for color variation
                        legend_left_bound = min(legend_left_bound, x_full)

        return legend_left_bound

    def _detect_plot_boundaries_by_content(self, plot_img):
        """Detect plot boundaries by analyzing content distribution"""
        h, w = plot_img.shape[:2]
        gray = cv2.cvtColor(plot_img, cv2.COLOR_BGR2GRAY)

        # Detect edges
        edges = cv2.Canny(gray, 30, 100)

        # Project edges onto axes to find boundaries
        horizontal_projection = np.sum(edges, axis=1)  # Sum along width (vertical projection)
        vertical_projection = np.sum(edges, axis=0)    # Sum along height (horizontal projection)

        # Find the main plot area by looking for high edge density
        # Smooth the projections to reduce noise
        def simple_smooth(data, window=5):
            """Simple moving average smoothing"""
            if len(data) < window:
                return data
            smoothed = np.convolve(data, np.ones(window)/window, mode='same')
            return smoothed

        smooth_h = simple_smooth(horizontal_projection)
        smooth_v = simple_smooth(vertical_projection)

        # Find boundaries based on edge density
        h_threshold = np.mean(smooth_h) + np.std(smooth_h) * 0.5
        v_threshold = np.mean(smooth_v) + np.std(smooth_v) * 0.5

        # Find top and bottom boundaries
        top_candidates = np.nonzero(smooth_h > h_threshold)[0]
        if len(top_candidates) > 0:
            plot_top = max(10, top_candidates[0] - 5)
            plot_bottom = min(h - 10, top_candidates[-1] + 5)
        else:
            plot_top = int(0.1 * h)
            plot_bottom = int(0.9 * h)

        # Find left and right boundaries
        left_candidates = np.nonzero(smooth_v > v_threshold)[0]
        if len(left_candidates) > 0:
            plot_left = max(10, left_candidates[0] - 5)
            # Don't use the rightmost edge for right boundary as it might include legends
            # Use only first 80% of candidates
            right_idx = int(len(left_candidates) * 0.8)
            plot_right = min(w - 10, left_candidates[right_idx] if right_idx < len(left_candidates) else left_candidates[-1])
        else:
            plot_left = int(0.15 * w)
            plot_right = int(0.75 * w)

        return {
            'plot_left': plot_left,
            'plot_right': plot_right,
            'plot_top': plot_top,
            'plot_bottom': plot_bottom
        }

    def _detect_plot_area(self, plot_img):
        """Detect the actual plot area excluding axis labels and legends"""
        h, w = plot_img.shape[:2]

        # Get default margins
        default_margins = self._get_default_margins(w, h)

        # Try content-based boundary detection
        try:
            content_bounds = self._detect_plot_boundaries_by_content(plot_img)
        except Exception:
            content_bounds = None

        # Try to detect axis lines (gracefully handle cv2 import issues)
        vertical_lines = []
        horizontal_lines = []
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

        # Detect legend area to adjust plot boundaries
        legend_left_bound = self._detect_legend_area_bounds(plot_img)

        # Combine different detection methods for more accurate boundaries
        if content_bounds:
            # Use content-based detection as primary, but ensure reasonable margins
            plot_left = max(left_margin, content_bounds['plot_left'])
            plot_top = max(default_margins['top'], content_bounds['plot_top'])
            plot_bottom = min(h - bottom_margin, content_bounds['plot_bottom'])

            # For right boundary, prefer legend detection if available
            if legend_left_bound < w * 0.9:
                plot_right = legend_left_bound - 10
            else:
                plot_right = min(content_bounds['plot_right'], w - default_margins['right'])
        else:
            # Fallback to margin-based detection
            plot_left = left_margin
            plot_top = default_margins['top']
            plot_bottom = h - bottom_margin

            if legend_left_bound < w * 0.9:
                plot_right = legend_left_bound - 10
            else:
                plot_right = w - default_margins['right']

        plot_area = {
            'plot_left': plot_left,
            'plot_right': plot_right,
            'plot_top': plot_top,
            'plot_bottom': plot_bottom,
            'full_width': w,
            'full_height': h
        }

        # Store detection results for visualization
        self._detection_results.update({
            'plot_area': plot_area,
            'vertical_lines': vertical_lines,
            'horizontal_lines': horizontal_lines
        })

        return plot_area

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

    def get_detection_results(self) -> Dict[str, Any]:
        """Get the intermediate detection results for visualization"""
        return self._detection_results.copy()

    def create_detection_visualization(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Create visualization of detection results"""
        if not self._detection_results:
            return {}

        return self._visualizer.create_step_by_step_visualization(image, self._detection_results)

    def create_detection_overview(self, image: np.ndarray) -> np.ndarray:
        """Create comprehensive detection overview"""
        if not self._detection_results:
            return image

        return self._visualizer.create_detection_overview(image, self._detection_results)

    def extract(self, image_path: str):
        img = ImageLoader.read(image_path)
        boxes = self._plot_detector.detect_plots(img)
        plots: List[PlotData] = []

        for pid, (x, y, w, h) in enumerate(boxes):
            crop = img[y:y+h, x:x+w]

            # Detect the actual plot area to avoid labels
            plot_area = self._detect_plot_area(crop)

            # Detect legend areas
            legend_boxes = self._legend_detector.detect_legend_boxes(crop)
            self._detection_results['legend_boxes'] = legend_boxes

            # Get calibration for this plot (use plot area for better accuracy)
            calibration = self._calibrate_axes_with_area(crop, plot_area)

            # Detect axis labels from strips left and bottom
            left_strip = crop[:, :int(0.15*w)]
            bottom_strip = crop[int(0.85*h):, :]
            y_label, y_unit = self._axis_label_detector.detect(left_strip)
            x_label, x_unit = self._axis_label_detector.detect(bottom_strip)

            # Extract markers in pixel coordinates
            raw_pts = self._marker_extractor.extract(crop)

            # Store raw data points for visualization
            self._detection_results['data_points'] = raw_pts

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
