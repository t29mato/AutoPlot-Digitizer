from __future__ import annotations
from typing import List
import numpy as np
from .image_loader import ImageLoader
from .plot_detector import PlotDetector
from .axis_label_detector import AxisLabelDetector
from .marker_extractor import MarkerExtractor
from ..domain.models import PlotData, Series, DigitizationResult, AxisInfo
from ..domain.repositories import IPlotExtractor

class OpenCVPlotExtractor(IPlotExtractor):
    def __init__(self):
        self._plot_detector = PlotDetector()
        self._axis_label_detector = AxisLabelDetector()
        self._marker_extractor = MarkerExtractor()

    def _calibrate_axes(self, plot_img):
        # naive: assume axes span full plot box; define transform using borders
        h, w = plot_img.shape[:2]
        return (0, w), (h, 0)  # pixel→data placeholder (will be refined per user config)

    def extract(self, image_path: str):
        img = ImageLoader.read(image_path)
        boxes = self._plot_detector.detect_plots(img)
        plots: List[PlotData] = []
        for pid, (x,y,w,h) in enumerate(boxes):
            crop = img[y:y+h, x:x+w]
            # detect axis labels from strips left and bottom
            left_strip = crop[:, :int(0.15*w)]
            bottom_strip = crop[int(0.85*h):, :]
            y_label, y_unit = self._axis_label_detector.detect(left_strip)
            x_label, x_unit = self._axis_label_detector.detect(bottom_strip)
            # marker extraction
            raw_pts = self._marker_extractor.extract(crop)
            # convert pixel→data → placeholder (requires calibration)
            # here we keep pixel coords; user can post‑process
            series_objs: List[Series] = [Series(sample=name, points=[(float(px), float(py)) for px,py in pts])
                                          for name, pts in raw_pts.items()]
            plots.append(PlotData(id=pid,
                                  x_axis=AxisInfo(label=x_label, unit=x_unit),
                                  y_axis=AxisInfo(label=y_label, unit=y_unit),
                                  series=series_objs))
        return DigitizationResult(plots=plots)
