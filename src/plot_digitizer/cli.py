import typer
from pathlib import Path
from .use_cases.extract_plots import ExtractPlotsUseCase
from .infrastructure.extractor_impl import OpenCVPlotExtractor

app = typer.Typer(help="Extract XY data from images of scientific plots → JSON")

@app.command()
def main(image: str, out: str = "output/output.json"):
    """image: path to PNG/JPEG/PDF page"""

    # Ensure output directory exists
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    use_case = ExtractPlotsUseCase(OpenCVPlotExtractor())
    use_case.execute(image, out)
    typer.echo(f"✅ Data written to {out}")

if __name__ == "__main__":
    app()
