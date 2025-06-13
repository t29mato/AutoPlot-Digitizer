from __future__ import annotations
import cv2
import pytesseract
import re
from typing import Tuple

class AxisLabelDetector:
    _regex = re.compile(r"([A-Za-zα-ωΑ-Ω]+)\s*\(?\s*([\w/\^μπ%]*)\s*\)?")

    def detect(self, axis_strip) -> Tuple[str,str|None]:
        """Return (label, unit) parsed from image strip."""
        gray = cv2.cvtColor(axis_strip, cv2.COLOR_BGR2GRAY)
        txt = pytesseract.image_to_string(gray, config="--psm 7")
        for token in txt.splitlines():
            m = self._regex.search(token)
            if m:
                lbl, unit = m.group(1), m.group(2) if m.group(2) else None
                return lbl, unit
        return "", None
