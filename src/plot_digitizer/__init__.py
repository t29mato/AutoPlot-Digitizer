"""Top‑level package for Plot‑Digitizer."""

from importlib import import_module
from typing import Dict, Type

from .interfaces.shape_detector import ShapeDetector

__all__ = [
    "register_shape_detector",
    "get_shape_detector",
]

_shape_registry: Dict[str, Type[ShapeDetector]] = {}

def register_shape_detector(name: str, cls: Type[ShapeDetector]):
    """Register a concrete ShapeDetector so the user can select it via CLI."""
    _shape_registry[name.lower()] = cls


def get_shape_detector(name: str) -> Type[ShapeDetector]:
    if name.lower() not in _shape_registry:
        raise KeyError(f"Unknown shape‑detector '{name}'. Registered: {list(_shape_registry)}")
    return _shape_registry[name.lower()]

# default detectors autoload
for _mod in [
    "plot_digitizer.infrastructure.opencv_shape_detector",
]:
    import_module(_mod)
