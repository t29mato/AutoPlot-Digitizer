from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Any
from .models import DigitizationResult

class IPlotExtractor(ABC):
    @abstractmethod
    def extract(self, image_path: str) -> DigitizationResult:  # pragma: no cover
        ...
