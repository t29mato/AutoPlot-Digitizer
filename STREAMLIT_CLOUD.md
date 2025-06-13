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

1. **streamlit_requirements.txt** - Python dependencies (cloud-optimized)
2. **packages.txt** - System packages (tesseract-ocr, OpenCV dependencies)
3. **ui/app.py** - Main Streamlit application
4. **.streamlit/config.toml** - Streamlit configuration

## Troubleshooting Common Issues

### 1. Connection Refused Error
If you see `dial tcp 127.0.0.1:8501: connect: connection refused`:
- Check that server.headless = true in config.toml
- Ensure serverAddress = "0.0.0.0" for cloud deployment

### 2. Import Errors
The app automatically:
- Adds src directory to Python path
- Uses opencv-python-headless for cloud compatibility
- Handles missing dependencies gracefully

### 3. OpenCV Issues
- Uses opencv-python-headless (not opencv-python)
- Sets OPENCV_IO_ENABLE_OPENEXR=0 environment variable
- Includes required system packages in packages.txt

## Environment Variables (Optional)

Set these in Streamlit Cloud if needed:
- `OPENCV_IO_ENABLE_OPENEXR=0`
- `PYTHONPATH=/app/src`

## Testing Locally

Before deploying, test with cloud-compatible setup:
```bash
# Install cloud requirements
pip install -r streamlit_requirements.txt

# Set environment variables
export OPENCV_IO_ENABLE_OPENEXR=0

# Run the app
streamlit run ui/app.py --server.address=0.0.0.0
```

## Deployment Checklist

- [ ] streamlit_requirements.txt uses opencv-python-headless
- [ ] packages.txt includes all system dependencies
- [ ] .streamlit/config.toml has headless=true
- [ ] ui/app.py handles import errors gracefully
- [ ] Local testing with cloud requirements passes
