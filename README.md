# AutoPlot-Digitizer

**Automated plot digitizer for scientific papers and research data extraction**

Extract **all XY data** from one or more 2‑D scatter/marker plots _inside the same image_ with full automation.

## 🎯 Features

* **Fully Automated** - No manual calibration required
* **Multi-plot Detection** - Automatically handles multiple sub-plots in one image
* **OCR Integration** - Extracts axis labels and units automatically
* **Scientific Focus** - Optimized for academic papers and research data
* **Clean Architecture** - Easy to extend and customize
* **Multiple Interfaces** - CLI, Python library, and Web UI (Streamlit)
* **Interactive Visualization** - Built-in plotting and data exploration

## 📦 Installation

```bash
pip install autoplot-digitizer
```

## 🚀 Quick Start

### Web UI (Streamlit)
```bash
# Install with UI dependencies
pip install autoplot-digitizer[ui]

# Run the web interface
python run_ui.py
```
Then open http://localhost:8501 in your browser for an intuitive drag-and-drop interface.

### CLI Usage
```bash
# Extract data from a plot image
autoplot-digitizer path/to/your/plot.png

# Specify output file
autoplot-digitizer plot.png --out results.json
```

### Python Library Usage
```python
from autoplot_digitizer.infrastructure.extractor_impl import OpenCVPlotExtractor
from autoplot_digitizer.use_cases.extract_plots import ExtractPlotsUseCase

# Initialize extractor and use case
extractor = OpenCVPlotExtractor()
use_case = ExtractPlotsUseCase(extractor)

# Extract data from plot
use_case.execute("path/to/plot.png", "output.json")

# Read extracted data
import json
with open("output.json", 'r') as f:
    data = json.load(f)

print(f"Found {len(data['plots'])} plots")
for plot in data['plots']:
    print(f"Plot {plot['id']}: {plot['x_axis']['label']} vs {plot['y_axis']['label']}")
```

## 📊 Output Format

JSON structure:
```json
{
  "plots": [
    {
      "id": 0,
      "x_axis": {"label": "Temperature", "unit": "K"},
      "y_axis": {"label": "Thermal Conductivity", "unit": "W/mK"},
      "series": [
        {
          "sample": "Sample-1",
          "points": [[300.0, 2.1], [350.0, 1.8], [400.0, 1.5]]
        },
        {
          "sample": "Sample-2",
          "points": [[300.0, 2.3], [350.0, 2.0], [400.0, 1.7]]
        }
      ]
    }
  ]
}
```

---
### Project layout
```
plot_digitizer/
├── domain/            # pure business objects (no I/O)
│   ├── models.py
│   └── repositories.py
├── use_cases/         # orchestration – application logic
│   └── extract_plots.py
├── infrastructure/    # frameworks, libs, ext. services
│   ├── image_loader.py
│   ├── plot_detector.py
│   ├── axis_label_detector.py
│   ├── marker_extractor.py
│   └── output_writer.py
└── cli.py             # CLI entry‑point (depends only on use_cases)
```

---
### Extending marker shapes
1. Sub‑class `MarkerExtractor` and register via `ENTRY_POINTS` (see `setup.cfg`).
2. Implement `extract(self, image_section) -> list[tuple[float,float]]` returning pixel coords.

---
### Roadmap
- PDF page iterator
- CUDA‑accelerated clustering
- GUI (PySide6)

## 🎓 Academic Use Cases

Perfect for:
- **Research Data Extraction** - Extract data from published papers
- **Meta-Analysis** - Collect data across multiple studies
- **Literature Review** - Digitize plots for comparison
- **Teaching** - Convert textbook graphs to interactive data

## 🤝 Contributing

Contributions welcome! This project uses:
- Clean Architecture for maintainability
- OpenCV for computer vision
- Tesseract OCR for text recognition
- Typer for CLI interface

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙋‍♂️ Author

Created by [t29mato](https://github.com/t29mato)
