"""Example value objects (scales, axes). Extend when needed."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LinearScale:
    pixel_start: float
    pixel_end: float
    value_start: float
    value_end: float

    def pixel_to_value(self, px: float) -> float:
        return (
            (px - self.pixel_start)
            / (self.pixel_end - self.pixel_start)
            * (self.value_end - self.value_start)
            + self.value_start
        )
