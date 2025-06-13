"""Thin CLI wrapper using Typer."""

import sys
from pathlib import Path

import typer

from plot_digitizer import get_shape_detector
from plot_digitizer.infrastructure.opencv_calibration import OpenCVAxisCalibrator
from plot_digitizer.infrastructure.opencv_color_mask import HSVColorMasker
from plot_digitizer.usecases.digitize import DigitizePlotUseCase

app = typer.Typer(add_completion=False)

_DEFAULT_HSV_WINDOWS = {
    "x = 2.00": ((0, 0, 0), (180, 255, 40)),  # black
    "x = 1.80": ((0, 150, 150), (10, 255, 255)),  # red
    "x = 1.60": ((110, 100, 120), (130, 255, 255)),  # blue
    "x = 1.52": ((85,  80, 120), (100, 255, 255)),  # teal
    "x = 1.40": ((145, 80, 120), (165, 255, 255)),  # magenta
}


@app.command()
def main(
    image: Path = typer.Argument(..., help="Path to plot image (png/jpg)."),
    out: Path = typer.Option("output.csv", help="CSV out path"),
    shape_set: str = typer.Option("default", help="Registered ShapeDetector name"),
):
    """Digitize a 2‑D scatter plot and dump CSV."""
    try:
        shape_cls = get_shape_detector(shape_set)
    except KeyError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)

    usecase = DigitizePlotUseCase(
        calibrator=OpenCVAxisCalibrator(),
        mask_maker=HSVColorMasker(_DEFAULT_HSV_WINDOWS),
        shape_detector=shape_cls(),
    )
    result = usecase.execute(image)
    usecase.to_csv(result, out)
    typer.echo(f"Extracted {len(result.datapoints)} points → {out}")


if __name__ == "__main__":
    app()
