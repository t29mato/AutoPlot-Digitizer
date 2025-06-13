# Plot‑Digitizer

Extract **all XY data** from one or more 2‑D scatter/marker plots _inside the same image_.

* Multiple sub‑plots automatically detected (matplotlib‑style grid or irregular layout)
* Axis **physical quantity / unit** parsed via OCR (pytesseract) → stored in JSON
* Clean‑Architecture: domain → use‑case → interface layers
* Easy to extend: plug‑in new `MarkerExtractor` / `PlotDetector` / `OutputWriter`

```bash
# 1. Install (python ≥3.9)
python -m pip install -r requirements.txt

# 2. Run (single image) and get JSON
python -m plot_digitizer.cli --image fig9.png --out data.json
```

JSON structure
```json
{
  "plots": [
    {
      "id": 0,
      "x_axis": {"label": "T", "unit": "K"},
      "y_axis": {"label": "kappa", "unit": "W/mK"},
      "series": [
        {"sample": "x=2.00", "points": [[20,4.3], [50,3.8], ...]},
        ...
      ]
    },
    ...
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
