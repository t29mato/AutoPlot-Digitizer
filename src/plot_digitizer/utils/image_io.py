from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return img


def save_debug(image: np.ndarray, path: str | Path):
    cv2.imwrite(str(path), image)
