from __future__ import annotations
import cv2

class ImageLoader:
    @staticmethod
    def read(path: str):
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(path)
        return image
