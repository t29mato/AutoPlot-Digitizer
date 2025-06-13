from __future__ import annotations
from typing import List, Tuple
from pydantic import BaseModel

class AxisInfo(BaseModel):
    label: str
    unit: str | None = None

class Series(BaseModel):
    sample: str
    points: List[Tuple[float, float]]  # (x, y)

class PlotData(BaseModel):
    id: int
    x_axis: AxisInfo
    y_axis: AxisInfo
    series: List[Series]

class DigitizationResult(BaseModel):
    plots: List[PlotData]
