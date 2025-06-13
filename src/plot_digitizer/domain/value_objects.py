"""Value objects for the plot digitizer domain."""

from dataclasses import dataclass


@dataclass(frozen=True)
class LinearScale:
    """Linear scale for converting between pixel coordinates and actual values."""

    pixel_start: float
    pixel_end: float
    value_start: float
    value_end: float

    def pixel_to_value(self, pixel: float) -> float:
        """Convert pixel coordinate to actual value."""
        if self.pixel_end == self.pixel_start:
            return self.value_start

        # Calculate the ratio along the pixel range
        pixel_ratio = (pixel - self.pixel_start) / (self.pixel_end - self.pixel_start)

        # Apply the ratio to the value range
        value = self.value_start + pixel_ratio * (self.value_end - self.value_start)

        return value

    def value_to_pixel(self, value: float) -> float:
        """Convert actual value to pixel coordinate."""
        if self.value_end == self.value_start:
            return self.pixel_start

        # Calculate the ratio along the value range
        value_ratio = (value - self.value_start) / (self.value_end - self.value_start)

        # Apply the ratio to the pixel range
        pixel = self.pixel_start + value_ratio * (self.pixel_end - self.pixel_start)

        return pixel
