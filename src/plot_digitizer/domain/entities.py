from dataclasses import dataclass
from typing import List


@dataclass(frozen=True, slots=True)
class DataPoint:
    """Physical data point after calibration."""

    x: float  # e.g. temperature (K)
    y: float  # e.g. kappa or ZT
    series: str  # e.g. "x = 1.80"


@dataclass
class PlotImage:
    """Raw plot + metadata container."""

    image_path: str
    # later we might store PIL.Image or np.ndarray, but keep path here for domain purity


@dataclass
class DigitizeResult:
    """Return object for use‑case."""

    datapoints: List[DataPoint]
