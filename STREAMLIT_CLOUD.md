# Streamlit Cloud Configuration

## Deployment Settings

When deploying to Streamlit Cloud, use these settings:

**Repository Settings:**
- Repository: `t29mato/autoplot-digitizer`
- Branch: `main` (or your preferred branch)
- Main file path: `ui/app.py`

**Advanced Settings:**
- Python version: `3.11` (or compatible)
- Requirements file: `streamlit_requirements.txt`
- Packages file: `packages.txt`

## Required Files

1. **streamlit_requirements.txt** - Python dependencies
2. **packages.txt** - System packages (tesseract-ocr, etc.)
3. **ui/app.py** - Main Streamlit application

## Environment Variables (Optional)

If needed, you can set these in Streamlit Cloud:
- `PYTHONPATH=/app/src` (if path issues persist)

## Known Issues & Solutions

1. **Import Issues**: The app automatically adds the src directory to Python path
2. **OpenCV Issues**: Using `opencv-python-headless` for cloud compatibility
3. **Tesseract Issues**: System packages are defined in `packages.txt`

## Testing Locally

Before deploying, test locally:
```bash
# Install cloud-compatible requirements
pip install -r streamlit_requirements.txt

# Run the app
streamlit run ui/app.py
```
