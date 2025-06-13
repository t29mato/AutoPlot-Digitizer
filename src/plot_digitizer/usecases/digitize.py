"""Application logic: orchestrates calibration → masking → detection → scaling."""

from pathlib import Path
from typing import List

import pandas as pd

from plot_digitizer.domain.entities import DataPoint, DigitizeResult
from plot_digitizer.interfaces.calibration import AxisCalibrator
from plot_digitizer.interfaces.color_mask import ColorMasker
from plot_digitizer.interfaces.shape_detector import ShapeDetector
from plot_digitizer.utils.image_io import read_image


class DigitizePlotUseCase:
    def __init__(
        self,
        calibrator: AxisCalibrator,
        mask_maker: ColorMasker,
        shape_detector: ShapeDetector,
    ):
        self.cal = calibrator
        self.masker = mask_maker
        self.detector = shape_detector

    def execute(self, image_path: str | Path) -> DigitizeResult:
        img = read_image(image_path)
        x_scale, y_scale = self.cal.calibrate(img)
        masks = self.masker.mask(img)

        datapoints: List[DataPoint] = []
        for series_name, mask in masks.items():
            centers = self.detector.detect(mask)
            for cx, cy in centers:
                datapoints.append(
                    DataPoint(x=x_scale.pixel_to_value(cx), y=y_scale.pixel_to_value(cy), series=series_name)
                )
        return DigitizeResult(datapoints=datapoints)

    # helper: dump to CSV
    def to_csv(self, result: DigitizeResult, out_path: str | Path):
        df = pd.DataFrame(
            [(d.series, d.x, d.y) for d in result.datapoints],
            columns=["Series", "X", "Y"],
        ).sort_values(["Series", "X"])
        df.to_csv(out_path, index=False)
