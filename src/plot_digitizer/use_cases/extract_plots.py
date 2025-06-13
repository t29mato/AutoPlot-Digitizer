from __future__ import annotations
from pathlib import Path
from ..domain.repositories import IPlotExtractor
from ..domain.models import DigitizationResult

class ExtractPlotsUseCase:
    def __init__(self, extractor: IPlotExtractor):
        self._extractor = extractor

    def execute(self, image_path: str, output_path: str):
        result: DigitizationResult = self._extractor.extract(image_path)
        Path(output_path).write_text(result.model_dump_json(indent=2), encoding="utf-8")
