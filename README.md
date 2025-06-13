# Plot‑Digitizer (Clean Architecture Edition)

Extract XY data points from static 2‑D scientific plots (PNG/JPEG/PDF) **automatically**
and **extensibly**.

- **Domain‑driven, Clean‑Architecture structure** – easy to swap algorithms.
- Out‑of‑the‑box support for colored scatter/marker plots (■ ● ▲ ▼ ◀, etc.).
- Plug‑in interface so you can add exotic glyphs (✦, ◆, Unicode, custom logos …).
- Pure‑Python (>=3.9) – only OpenCV, NumPy, scikit‑image, scikit‑learn & pandas.
- CLI & Python API; outputs tidy CSV.

```bash
# Quickstart (assumes venv is activated)
$ pip install -r requirements.txt
$ python -m plot_digitizer.cli ./fig9.png --out kappa_T.csv --shape-set default
```

---
## Project layout
```
plot_digitizer_repo/
├─ README.md
├─ requirements.txt
└─ src/
   └─ plot_digitizer/
      ├─ __init__.py
      ├─ domain/
      │   ├─ entities.py
      │   └─ value_objects.py
      ├─ usecases/
      │   └─ digitize.py
      ├─ interfaces/
      │   ├─ calibration.py
      │   ├─ color_mask.py
      │   └─ shape_detector.py
      ├─ infrastructure/
      │   ├─ opencv_calibration.py
      │   ├─ opencv_color_mask.py
      │   └─ opencv_shape_detector.py
      ├─ utils/
      │   ├─ clustering.py
      │   └─ image_io.py
      └─ cli.py
```

### Key architectural rules
1. **Domain layer** knows nothing about OpenCV/sk‑image, only plain Python objects.
2. **Use‑case layer** orchestrates domain logic – no image‑processing specifics.
3. **Interface layer** defines *ports* (abstract base classes) – e.g. `ShapeDetector`.
4. **Infrastructure layer** provides concrete adapters that depend on external libs.
5. CLI / Presentation imports *only* use‑cases.

### Extending to new symbols
Implement a subclass of `ShapeDetector` that overrides `detect_markers()`
and register it via `plot_digitizer.register_shape_detector("my_symbol", MyDetector)`.

---
## License
MIT
