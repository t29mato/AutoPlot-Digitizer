# AutoPlot-Digitizer Web UI

This directory contains the Streamlit-based web interface for AutoPlot-Digitizer.

## 🚀 Quick Start

### Local Development
1. **Install dependencies:**
   ```bash
   pip install autoplot-digitizer[ui]
   ```

2. **Run the UI:**
   ```bash
   # From the project root directory
   streamlit run ui/app.py
   ```

3. **Open in browser:**
   - The app will automatically open at `http://localhost:8501`
   - Or use the convenience script: `python run_ui.py`

### Streamlit Cloud Deployment

1. **Fork the repository** on GitHub
2. **Connect to Streamlit Cloud** at [share.streamlit.io](https://share.streamlit.io)
3. **Deploy settings:**
   - **Main file path:** `ui/app.py`
   - **Requirements file:** `streamlit_requirements.txt`
   - **Packages file:** `packages.txt` (for system dependencies)
4. **Click Deploy!**

**Required files for Streamlit Cloud:**
- `streamlit_requirements.txt` - Python dependencies
- `packages.txt` - System packages (tesseract-ocr, etc.)
- `ui/app.py` - Main application file

## 📋 Features

- **Drag & Drop Interface** - Easy file upload
- **Real-time Processing** - Live extraction results
- **Interactive Visualization** - Plot the extracted data
- **Multiple Export Formats** - JSON, CSV downloads
- **Multi-plot Support** - Handle multiple plots in one image

## 🔧 Configuration

Configuration is handled through `.streamlit/config.toml`:
- Upload size limit: 200MB
- Default port: 8501
- Theme: Custom color scheme

## 📁 Files

- `app.py` - Main Streamlit application
- `test_app.py` - Simple test version
- `../run_ui.py` - Convenience runner script
- `../.streamlit/config.toml` - Streamlit configuration

## 🎯 Usage Guide

1. **Upload Plot Image** - Drag and drop or browse for your scientific plot
2. **Choose Output Format** - Select JSON, CSV, or both
3. **Extract Data** - Click the extract button to process
4. **Review Results** - View the extracted data and visualization
5. **Download** - Save the results in your preferred format

## 🛠️ Development

To modify the UI:
1. Edit `app.py` for main functionality
2. Modify `.streamlit/config.toml` for configuration
3. Update styling in the CSS section of `app.py`

## 📊 Supported Formats

**Input:**
- PNG, JPEG images
- PDF files (first page only)

**Output:**
- JSON (structured data)
- CSV (tabular format)
- Interactive plots (Matplotlib)
