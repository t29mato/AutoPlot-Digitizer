from __future__ import annotations
from pathlib import Path
from ..domain.models import DigitizationResult

class JSONWriter:
    def write(self, result: DigitizationResult, path: str):
        Path(path).write_text(result.model_dump_json(indent=2), encoding="utf-8")
