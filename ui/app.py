"""
AutoPlot-Digitizer Streamlit Web UI

A web interface for automated plot digitization from scientific papers.
"""

import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import tempfile
import os
from PIL import Image
import io

# Add the src directory to the path for imports
import sys
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent
src_path = project_root / "src"

# Add src to Python path if not already present
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Fix OpenCV issues in cloud environments
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'

try:
    # Test OpenCV import and fix common issues
    import cv2
    # Test basic functionality
    _ = cv2.__version__

    from plot_digitizer.infrastructure.extractor_impl import OpenCVPlotExtractor
    from plot_digitizer.use_cases.extract_plots import ExtractPlotsUseCase
    from plot_digitizer.domain.models import DigitizationResult
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please make sure the plot_digitizer package is properly installed.")
    st.error("For Streamlit Cloud, make sure you're using opencv-python-headless.")
    st.stop()
except Exception as e:
    st.error(f"OpenCV initialization error: {e}")
    st.error("This might be a system-level dependency issue.")
    st.stop()

# Page config
st.set_page_config(
    page_title="AutoPlot-Digitizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Header
    st.title("📊 AutoPlot-Digitizer")
    st.markdown("**Automated plot digitizer for scientific papers and research data extraction**")

    # Show system info in debug mode
    if st.session_state.get('debug_mode', False):
        import platform
        st.sidebar.write(f"Python: {platform.python_version()}")
        st.sidebar.write(f"Platform: {platform.system()}")
        try:
            import cv2
            st.sidebar.write(f"OpenCV: {cv2.__version__}")
        except ImportError:
            st.sidebar.write("OpenCV: Not available")

    # Sidebar
    with st.sidebar:
        st.header("🎯 Features")
        st.markdown("""
        ✅ **Fully Automated** - No manual calibration required
        ✅ **Multi-plot Detection** - Multiple sub-plots in one image
        ✅ **OCR Integration** - Automatic axis labels extraction
        ✅ **Scientific Focus** - Optimized for research data
        ✅ **Clean Output** - Structured JSON results
        """)

        st.header("📚 Perfect for")
        st.markdown("""
        - Research Data Extraction
        - Meta-Analysis Studies
        - Literature Review
        - Teaching & Education
        """)

        st.header("ℹ️ About")
        st.markdown("""
        Created by [t29mato](https://github.com/t29mato)
        MIT License
        """)

        # Debug mode (for development)
        st.header("🔧 Debug")
        st.session_state['debug_mode'] = st.checkbox("Enable debug mode", False)

        if st.button("🔄 Clear Cache & Reload"):
            st.cache_data.clear()
            st.rerun()

        # Module debug info
        if st.session_state.get('debug_mode', False):
            try:
                from plot_digitizer.infrastructure.extractor_impl import OpenCVPlotExtractor
                test_extractor = OpenCVPlotExtractor()
                st.write(f"✅ Extractor import: OK")
                st.write(f"✅ Has get_detection_results: {hasattr(test_extractor, 'get_detection_results')}")
                st.write(f"✅ Available methods: {[m for m in dir(test_extractor) if 'detection' in m]}")
            except Exception as e:
                st.write(f"❌ Import error: {e}")

    # Main interface
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Extract Data", "� Detection Analysis", "�📊 View Results", "💡 Examples"])

    with tab1:
        extract_data_tab()

    with tab2:
        detection_analysis_tab()

    with tab3:
        view_results_tab()

    with tab4:
        examples_tab()

def extract_data_tab():
    st.header("Upload and Process Your Plot")

    # File upload
    uploaded_file = st.file_uploader(
        "📎 Drop your graph image here or click to browse",
        type=['png', 'jpg', 'jpeg', 'pdf'],
        help="Supported formats: PNG, JPG, JPEG, PDF. Processing starts automatically after upload."
    )

    if uploaded_file is not None:
        # Create 2 columns for original and result
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 Original Image")
            if uploaded_file.type.startswith('image'):
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Plot", use_column_width=True)
            else:
                st.info("PDF uploaded - will process first page")

        with col2:
            st.subheader("🎯 Extraction Results")

            # Progress indicator
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Automatically start processing
            try:
                status_text.text("🔍 Analyzing image...")
                progress_bar.progress(25)

                # Show debug info if enabled
                if st.session_state.get('debug_mode', False):
                    st.write(f"📁 File name: {uploaded_file.name}")
                    st.write(f"� File size: {len(uploaded_file.getvalue())} bytes")

                status_text.text("📊 Detecting plots...")
                progress_bar.progress(50)

                # Process the image
                results = process_plot(uploaded_file)

                status_text.text("🎨 Extracting data points...")
                progress_bar.progress(75)

                if results:
                    status_text.text("✨ Creating visualization...")
                    progress_bar.progress(90)

                    # Initialize session state for downloads if not exists
                    if 'download_state' not in st.session_state:
                        st.session_state['download_state'] = {}

                    # Display extraction results visualization
                    display_extraction_visualization(results)

                    progress_bar.progress(100)
                    status_text.text("✅ Extraction completed!")

                    # Store results in session state
                    st.session_state['extraction_results'] = results
                    st.session_state['uploaded_filename'] = uploaded_file.name

                    # Generate unique session key for this extraction
                    import time
                    st.session_state['current_extraction_key'] = f"{int(time.time() * 1000)}_{hash(uploaded_file.name)}"
                else:
                    st.warning("⚠️ No plots detected in the image. Please try with a clearer graph image.")
                    progress_bar.progress(100)
                    status_text.text("❌ No data extracted")

            except Exception as e:
                st.error(f"❌ Error during extraction: {str(e)}")
                progress_bar.progress(100)
                status_text.text("❌ Extraction failed")
                if st.session_state.get('debug_mode', False):
                    st.exception(e)

        # Display detailed results below if extraction was successful
        if 'extraction_results' in st.session_state and st.session_state['extraction_results']:
            # Show debug info if enabled
            if st.session_state.get('debug_mode', False):
                st.write(f"📁 File name: {uploaded_file.name}")
                st.write(f"📏 File size: {len(uploaded_file.read())} bytes")
                uploaded_file.seek(0)  # Reset file pointer

            with st.spinner("🔍 Analyzing plot structure..."):
                results = process_plot(uploaded_file)

            if results:
                st.success("✅ Data extraction completed!")

                # Store results in session state
                st.session_state['extraction_results'] = results
                st.session_state['uploaded_filename'] = uploaded_file.name

                # Display quick summary
                display_quick_summary(results)

                st.markdown("---")
                st.subheader("📋 Extraction Details")

                # Display detailed results
                display_detailed_results(st.session_state['extraction_results'])

                # Download buttons
                create_download_buttons(st.session_state['extraction_results'], st.session_state['uploaded_filename'])

    else:
        # Show help when no file is uploaded
        st.info("👆 Please upload a graph image to start the automatic extraction process")

        # Usage instructions
        with st.expander("📖 How to use"):
            st.markdown("""
            1. **Upload Image**: Click the file uploader above and select a graph image
            2. **Automatic Processing**: The system will automatically:
               - Detect plots in your image
               - Extract data points from each plot
               - Calibrate coordinates to actual axis values
               - Display results alongside the original image
            3. **View Results**: Check the extracted points and download as JSON/CSV

            **Supported formats**: PNG, JPG, JPEG, PDF

            **Best results with**:
            - Clear, high-contrast graphs
            - Visible axis labels and grid lines
            - Distinct colors for different data series
            """)

def display_detailed_results(results):
    """Display detailed extraction results"""
    for plot_idx, plot in enumerate(results['plots']):
        with st.expander(f"📊 Plot {plot_idx} Details", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Plot Information:**")
                st.write(f"- X-axis: {plot['x_axis']['label'] or 'Not detected'}")
                st.write(f"- Y-axis: {plot['y_axis']['label'] or 'Not detected'}")
                st.write(f"- Total Series: {len(plot['series'])}")
                st.write(f"- Total Points: {sum(len(s['points']) for s in plot['series'])}")

            with col2:
                # Color/series breakdown
                st.write("**Series Breakdown:**")
                for series in plot['series']:
                    point_count = len(series['points'])
                    st.write(f"- {series['sample']}: {point_count} points")

            # Sample points preview
            if plot['series']:
                st.write("**Sample Points (Calibrated Coordinates):**")
                sample_data = []

                for series in plot['series'][:3]:  # Show first 3 series
                    for i, point in enumerate(series['points'][:3]):  # Show first 3 points
                        sample_data.append({
                            'Series': series['sample'],
                            'Point': i + 1,
                            'X': round(point[0], 3),
                            'Y': round(point[1], 3)
                        })

                if sample_data:
                    df = pd.DataFrame(sample_data)
                    st.dataframe(df, use_container_width=True)

def create_temp_file_from_upload(uploaded_file):
    """Create a temporary file from uploaded file"""
    # Reset file pointer to beginning
    uploaded_file.seek(0)

    # Validate file type and get extension
    file_extension = uploaded_file.name.split('.')[-1].lower()
    allowed_extensions = ['png', 'jpg', 'jpeg', 'pdf']

    if file_extension not in allowed_extensions:
        raise ValueError(f"Unsupported file type: {file_extension}. Allowed: {allowed_extensions}")

    # Create temporary input file
    suffix = f".{file_extension}"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)

    # Write uploaded file data to temporary file
    file_data = uploaded_file.read()
    if len(file_data) == 0:
        raise ValueError("Uploaded file is empty")

    with os.fdopen(tmp_fd, 'wb') as tmp_file:
        tmp_file.write(file_data)

    # Verify the temporary file exists and has content
    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
        raise ValueError("Temporary file creation failed")

    return tmp_path

def cleanup_temp_files(*paths):
    """Clean up temporary files"""
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass  # Ignore cleanup errors

def process_plot(uploaded_file):
    """Process the uploaded plot file using new architecture"""
    tmp_path = None

    try:
        # Create temporary file
        tmp_path = create_temp_file_from_upload(uploaded_file)

        # Force module reload to ensure we have the latest version
        import importlib
        import plot_digitizer.infrastructure.extractor_impl
        importlib.reload(plot_digitizer.infrastructure.extractor_impl)

        # Process the image with calibrated coordinates (UI version uses calibrated)
        extractor = OpenCVPlotExtractor(x_range=(0, 600), y_range=(0, 6))

        # Execute extraction directly
        results = extractor.extract(tmp_path)

        # Get detection results for visualization
        try:
            detection_results = extractor.get_detection_results()
        except AttributeError as e:
            st.error(f"Detection results method not available: {e}")
            # Fallback - continue without visualization
            detection_results = {}

        # Create visualization images
        import cv2
        original_image = cv2.imread(tmp_path)
        if original_image is not None and detection_results:
            try:
                detection_visualization = extractor.create_detection_visualization(original_image)
                detection_overview = extractor.create_detection_overview(original_image)

                # Store in session state for the analysis tab
                st.session_state['detection_results'] = detection_results
                st.session_state['detection_visualization'] = detection_visualization
                st.session_state['detection_overview'] = detection_overview
            except Exception as viz_e:
                st.warning(f"Could not create visualizations: {viz_e}")
                # Continue without visualization
                if st.session_state.get('debug_mode', False):
                    st.exception(viz_e)

        # Convert to dictionary format for JSON serialization
        results_dict = {
            "plots": []
        }

        for plot in results.plots:
            plot_dict = {
                "id": plot.id,
                "x_axis": {
                    "label": plot.x_axis.label,
                    "unit": plot.x_axis.unit
                },
                "y_axis": {
                    "label": plot.y_axis.label,
                    "unit": plot.y_axis.unit
                },
                "series": []
            }

            for series in plot.series:
                series_dict = {
                    "sample": series.sample,
                    "points": series.points
                }
                plot_dict["series"].append(series_dict)

            results_dict["plots"].append(plot_dict)

        return results_dict

    except Exception as e:
        st.error(f"Error processing plot: {str(e)}")
        # Additional debug info for development
        if st.session_state.get('debug_mode', False):
            st.exception(e)
            if tmp_path:
                st.write(f"Temp file path: {tmp_path}")
                st.write(f"Temp file exists: {os.path.exists(tmp_path)}")
        return None

    finally:
        # Clean up temporary files
        cleanup_temp_files(tmp_path)

def detection_analysis_tab():
    """Tab for analyzing detection results step by step"""
    st.header("🔍 Detection Analysis")

    # Check if we have recent extraction results with detection data
    if ('extraction_results' not in st.session_state or
        'detection_visualization' not in st.session_state):
        st.info("👆 Upload and process a plot in the 'Extract Data' tab first to see detection analysis!")
        return

    st.markdown("""
    This analysis shows the step-by-step detection process to help you understand
    how the system identified different elements in your graph.
    """)

    detection_viz = st.session_state['detection_visualization']

    # Create tabs for different detection steps
    step_tabs = st.tabs([
        "1️⃣ Original",
        "2️⃣ Plot Area",
        "3️⃣ Axis Detection",
        "4️⃣ Data Points",
        "5️⃣ Complete Overview"
    ])

    step_names = ['1_original', '2_plot_area', '3_axis_detection', '4_data_points', '5_complete']
    step_descriptions = [
        "Original uploaded image",
        "Detected plot area boundaries (green rectangle shows the actual data plotting region)",
        "Detected axis lines (blue for Y-axis, red for X-axis)",
        "Extracted data points (different colors for different data series)",
        "Complete overview with all detected elements"
    ]

    for i, (tab, step_key, description) in enumerate(zip(step_tabs, step_names, step_descriptions)):
        with tab:
            st.markdown(f"**{description}**")

            if step_key in detection_viz:
                # Convert BGR to RGB for display
                viz_image = detection_viz[step_key]
                if len(viz_image.shape) == 3 and viz_image.shape[2] == 3:
                    viz_image_rgb = cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB)
                else:
                    viz_image_rgb = viz_image

                st.image(viz_image_rgb, caption=f"Step {i+1}: {description}", use_column_width=True)

                # Add specific information for each step
                detection_results = st.session_state.get('detection_results', {})

                if step_key == '2_plot_area' and 'plot_area' in detection_results:
                    plot_area = detection_results['plot_area']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Plot Width", f"{plot_area['plot_right'] - plot_area['plot_left']} px")
                        st.metric("Plot Height", f"{plot_area['plot_bottom'] - plot_area['plot_top']} px")
                    with col2:
                        st.metric("Left Margin", f"{plot_area['plot_left']} px")
                        st.metric("Bottom Margin", f"{plot_area['full_height'] - plot_area['plot_bottom']} px")

                elif step_key == '3_axis_detection':
                    v_lines = detection_results.get('vertical_lines', [])
                    h_lines = detection_results.get('horizontal_lines', [])
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Vertical Lines Detected", len(v_lines))
                        if v_lines:
                            st.write("Positions:", v_lines)
                    with col2:
                        st.metric("Horizontal Lines Detected", len(h_lines))
                        if h_lines:
                            st.write("Positions:", h_lines)

                elif step_key == '4_data_points' and 'data_points' in detection_results:
                    data_points = detection_results['data_points']
                    st.subheader("📊 Data Points Summary")

                    for series_name, points in data_points.items():
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric(f"Series: {series_name}", f"{len(points)} points")
                        with col2:
                            if len(points) > 0:
                                x_coords = [p[0] for p in points]
                                y_coords = [p[1] for p in points]
                                st.write(f"X range: {min(x_coords):.1f} - {max(x_coords):.1f} px")
                                st.write(f"Y range: {min(y_coords):.1f} - {max(y_coords):.1f} px")
            else:
                st.warning(f"Visualization for step {i+1} not available")

def display_quick_summary(results):
    """Display a quick summary of extraction results"""
    st.subheader("📈 Extraction Summary")

    total_plots = len(results['plots'])
    total_series = sum(len(plot['series']) for plot in results['plots'])
    total_points = sum(len(series['points']) for plot in results['plots'] for series in plot['series'])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Plots Detected", total_plots, help="Number of separate plots found")

    with col2:
        st.metric("Data Series", total_series, help="Number of data series across all plots")

    with col3:
        st.metric("Data Points", total_points, help="Total number of extracted points")

def create_download_buttons(results, filename):
    """Create download buttons for the results"""
    import time

    st.subheader("💾 Download Results")

    col1, col2 = st.columns(2)

    # Use session state extraction key if available, otherwise create unique key
    session_key = st.session_state.get('current_extraction_key', f"{int(time.time() * 1000)}_{hash(filename)}")

    # JSON download
    with col1:
        json_str = json.dumps(results, indent=2)
        st.download_button(
            label="📄 Download JSON",
            data=json_str,
            file_name=f"{Path(filename).stem}_extracted.json",
            mime="application/json",
            use_container_width=True,
            key=f"download_json_{session_key}"
        )

    # CSV download
    with col2:
        csv_data = convert_to_csv(results)
        st.download_button(
            label="📊 Download CSV",
            data=csv_data,
            file_name=f"{Path(filename).stem}_extracted.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"download_csv_{session_key}"
        )

def convert_to_csv(results):
    """Convert JSON results to CSV format"""
    csv_data = []

    for plot in results['plots']:
        for series in plot['series']:
            for point in series['points']:
                csv_data.append({
                    'Plot_ID': plot['id'],
                    'Series': series['sample'],
                    'X': point[0],
                    'Y': point[1],
                    'X_Label': plot['x_axis']['label'],
                    'X_Unit': plot['x_axis']['unit'],
                    'Y_Label': plot['y_axis']['label'],
                    'Y_Unit': plot['y_axis']['unit']
                })

    df = pd.DataFrame(csv_data)
    return df.to_csv(index=False)

def view_results_tab():
    """Tab for viewing and analyzing results"""
    st.header("View Extraction Results")

    if 'extraction_results' not in st.session_state:
        st.info("👆 Upload and process a plot in the 'Extract Data' tab first!")
        return

    results = st.session_state['extraction_results']
    filename = st.session_state.get('uploaded_filename', 'plot')

    st.subheader(f"📋 Results for: {filename}")

    # Plot selection
    plot_options = [f"Plot {plot['id']}: {plot['x_axis']['label']} vs {plot['y_axis']['label']}"
                   for plot in results['plots']]

    if len(plot_options) == 0:
        st.warning("No plots found in the results.")
        return

    selected_plot_idx = st.selectbox("Select plot to view:", range(len(plot_options)),
                                    format_func=lambda x: plot_options[x])

    plot_data = results['plots'][selected_plot_idx]

    # Display plot details
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Plot Information")
        st.write(f"**X-axis:** {plot_data['x_axis']['label']}")
        if plot_data['x_axis']['unit']:
            st.write(f"**X-unit:** {plot_data['x_axis']['unit']}")
        st.write(f"**Y-axis:** {plot_data['y_axis']['label']}")
        if plot_data['y_axis']['unit']:
            st.write(f"**Y-unit:** {plot_data['y_axis']['unit']}")
        st.write(f"**Series count:** {len(plot_data['series'])}")

    with col2:
        st.subheader("📈 Data Preview")
        total_points = sum(len(series['points']) for series in plot_data['series'])
        st.metric("Total Points", total_points)

        # Series breakdown
        for i, series in enumerate(plot_data['series'][:5]):  # Show first 5 series
            st.write(f"**{series['sample']}:** {len(series['points'])} points")

    # Visualization options and controls
    st.subheader("📈 Data Visualization")

    # Visualization controls
    viz_col1, viz_col2, viz_col3 = st.columns(3)

    with viz_col1:
        plot_type = st.selectbox(
            "Plot Type",
            ["Scatter Plot", "Line Plot", "Both (Scatter + Line)"],
            help="Choose how to display the data"
        )

    with viz_col2:
        show_grid = st.checkbox("Show Grid", True)

    with viz_col3:
        show_legend = st.checkbox("Show Legend", True)

    # Advanced options in an expander
    with st.expander("🎨 Advanced Plot Options"):
        adv_col1, adv_col2 = st.columns(2)

        with adv_col1:
            figure_size = st.selectbox(
                "Figure Size",
                ["Small (8x6)", "Medium (10x8)", "Large (12x9)", "Extra Large (14x10)"],
                index=1
            )

            marker_size = st.slider("Marker Size", 10, 100, 50)

        with adv_col2:
            line_width = st.slider("Line Width", 0.5, 5.0, 2.0, 0.5)
            alpha = st.slider("Transparency", 0.1, 1.0, 0.7, 0.1)

    # Create the visualization
    create_enhanced_plot_visualization(plot_data, plot_type, show_grid, show_legend,
                                     figure_size, marker_size, line_width, alpha)

    # Data statistics
    st.subheader("📊 Data Statistics")
    create_data_statistics(plot_data)

    # Data table
    st.subheader("📋 Data Table")
    create_data_table(plot_data)

def create_enhanced_plot_visualization(plot_data, plot_type, show_grid, show_legend,
                                     figure_size, marker_size, line_width, alpha):
    """Create enhanced matplotlib visualization with customizable options"""

    # Parse figure size
    size_map = {
        "Small (8x6)": (8, 6),
        "Medium (10x8)": (10, 8),
        "Large (12x9)": (12, 9),
        "Extra Large (14x10)": (14, 10)
    }
    figsize = size_map[figure_size]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Generate colors for series
    n_series = len(plot_data['series'])
    if n_series > 0:
        colors = plt.cm.tab10(np.linspace(0, 1, min(n_series, 10)))
        if n_series > 10:
            # Use additional color maps for more than 10 series
            colors2 = plt.cm.Set3(np.linspace(0, 1, n_series - 10))
            colors = np.vstack([colors, colors2])

    # Plot each series
    for i, series in enumerate(plot_data['series']):
        if not series['points']:
            continue

        x_vals = [point[0] for point in series['points']]
        y_vals = [point[1] for point in series['points']]

        color = colors[i % len(colors)]
        label = series['sample']

        # Choose plot type
        if plot_type == "Scatter Plot":
            ax.scatter(x_vals, y_vals, label=label, color=color,
                      alpha=alpha, s=marker_size, edgecolors='black', linewidth=0.5)
        elif plot_type == "Line Plot":
            # Sort points by x-value for line plots
            sorted_points = sorted(zip(x_vals, y_vals))
            x_sorted, y_sorted = zip(*sorted_points) if sorted_points else ([], [])
            ax.plot(x_sorted, y_sorted, label=label, color=color,
                   alpha=alpha, linewidth=line_width, marker='o', markersize=marker_size/10)
        else:  # Both (Scatter + Line)
            # Plot line first (underneath)
            sorted_points = sorted(zip(x_vals, y_vals))
            x_sorted, y_sorted = zip(*sorted_points) if sorted_points else ([], [])
            ax.plot(x_sorted, y_sorted, color=color, alpha=alpha*0.6, linewidth=line_width)
            # Then scatter points on top
            ax.scatter(x_vals, y_vals, label=label, color=color,
                      alpha=alpha, s=marker_size, edgecolors='black', linewidth=0.5)

    # Customize the plot
    x_label = plot_data['x_axis']['label'] or 'X'
    y_label = plot_data['y_axis']['label'] or 'Y'
    x_unit = plot_data['x_axis']['unit']
    y_unit = plot_data['y_axis']['unit']

    x_label_full = f"{x_label} ({x_unit})" if x_unit else x_label
    y_label_full = f"{y_label} ({y_unit})" if y_unit else y_label

    ax.set_xlabel(x_label_full, fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label_full, fontsize=12, fontweight='bold')
    ax.set_title(f"Extracted Plot Data: {y_label} vs {x_label}", fontsize=14, fontweight='bold')

    # Grid
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle='--')

    # Legend
    if show_legend and len(plot_data['series']) > 0:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    plt.tight_layout()
    st.pyplot(fig)

    # Add download option for the plot
    if st.button("💾 Download Plot as PNG", key="download_plot"):
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buf.seek(0)

        import time
        timestamp = int(time.time() * 1000)
        st.download_button(
            label="📸 Save Plot Image",
            data=buf.getvalue(),
            file_name=f"extracted_plot_{plot_data.get('id', 0)}.png",
            mime="image/png",
            key=f"download_plot_image_{plot_data.get('id', 0)}_{timestamp}"
        )
        plt.close(fig)
    else:
        plt.close(fig)

def create_plot_visualization(plot_data):
    """Create matplotlib visualization of the extracted data (legacy function)"""
    # Keep for backward compatibility, but use enhanced version
    create_enhanced_plot_visualization(plot_data, "Scatter Plot", True, True,
                                     "Medium (10x8)", 50, 2.0, 0.7)

def create_data_statistics(plot_data):
    """Create statistics summary for the extracted data"""
    if not plot_data['series']:
        st.warning("No data series found.")
        return

    # Calculate statistics for each series
    stats_data = []

    for series in plot_data['series']:
        if not series['points']:
            continue

        x_vals = [point[0] for point in series['points']]
        y_vals = [point[1] for point in series['points']]

        stats_data.append({
            'Series': series['sample'],
            'Points': len(series['points']),
            'X Min': min(x_vals),
            'X Max': max(x_vals),
            'X Mean': sum(x_vals) / len(x_vals),
            'Y Min': min(y_vals),
            'Y Max': max(y_vals),
            'Y Mean': sum(y_vals) / len(y_vals)
        })

    if stats_data:
        # Display as table
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df.round(3), use_container_width=True)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        total_points = sum(stat['Points'] for stat in stats_data)
        total_series = len(stats_data)

        with col1:
            st.metric("Total Series", total_series)
        with col2:
            st.metric("Total Points", total_points)
        with col3:
            x_range = max(stat['X Max'] for stat in stats_data) - min(stat['X Min'] for stat in stats_data)
            st.metric("X Range", f"{x_range:.2f}")
        with col4:
            y_range = max(stat['Y Max'] for stat in stats_data) - min(stat['Y Min'] for stat in stats_data)
            st.metric("Y Range", f"{y_range:.2f}")

def create_data_table(plot_data):
    """Create a data table from the plot data"""
    table_data = []

    for series in plot_data['series']:
        for point in series['points']:
            table_data.append({
                'Series': series['sample'],
                'X': round(point[0], 3),
                'Y': round(point[1], 3)
            })

    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, height=400)
    else:
        st.warning("No data points found in the selected plot.")

def examples_tab():
    """Tab showing examples and usage instructions"""
    st.header("💡 Examples & Usage Guide")

    # Example workflow
    st.subheader("🔄 Typical Workflow")

    workflow_steps = [
        "📤 **Upload** your scientific plot image (PNG, JPG, PDF)",
        "⚙️ **Configure** output format (JSON, CSV, or both)",
        "🔄 **Process** - Our AI automatically detects plots and extracts data",
        "📊 **Review** results in the visualization tab",
        "💾 **Download** extracted data in your preferred format"
    ]

    for step in workflow_steps:
        st.markdown(f"{step}")

    # Supported formats
    st.subheader("📁 Supported Input Formats")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Image Formats:**
        - PNG (.png)
        - JPEG (.jpg, .jpeg)
        - PDF (.pdf) - first page only
        """)

    with col2:
        st.markdown("""
        **Plot Types:**
        - Scatter plots
        - Line plots
        - XY plots
        - Multi-series plots
        - Log-scale plots
        """)

    # Best practices
    st.subheader("✨ Best Practices")

    best_practices = [
        "**High Resolution:** Use high-quality images (300+ DPI) for better accuracy",
        "**Clear Contrast:** Ensure good contrast between data points and background",
        "**Minimal Clutter:** Remove unnecessary annotations or watermarks if possible",
        "**Multiple Plots:** The tool can handle multiple sub-plots in one image",
        "**Axis Labels:** Clear, readable axis labels improve automatic detection"
    ]

    for practice in best_practices:
        st.markdown(f"• {practice}")

    # Output format explanation
    st.subheader("📋 Output Format")

    st.markdown("**JSON Structure:**")
    st.code("""
{
  "plots": [
    {
      "id": 0,
      "x_axis": {"label": "Temperature", "unit": "K"},
      "y_axis": {"label": "Thermal Conductivity", "unit": "W/mK"},
      "series": [
        {
          "sample": "Sample-1",
          "points": [[300.0, 2.1], [350.0, 1.8]]
        }
      ]
    }
  ]
}
    """, language="json")

    st.markdown("**CSV Structure:**")
    st.code("""
Plot_ID,Series,X,Y,X_Label,X_Unit,Y_Label,Y_Unit
0,Sample-1,300.0,2.1,Temperature,K,Thermal Conductivity,W/mK
0,Sample-1,350.0,1.8,Temperature,K,Thermal Conductivity,W/mK
    """, language="csv")

def display_extraction_visualization(results):
    """Display extraction results visualization"""
    if not results or 'plots' not in results:
        st.warning("No visualization data available")
        return

    # Create visualization for each plot
    for plot_idx, plot in enumerate(results['plots']):
        if plot['series']:
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot each series
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

            for series_idx, series in enumerate(plot['series']):
                if series['points']:
                    x_coords = [point[0] for point in series['points']]
                    y_coords = [point[1] for point in series['points']]

                    color = colors[series_idx % len(colors)]
                    ax.scatter(x_coords, y_coords,
                             c=color, alpha=0.7, s=30,
                             label=f"{series['sample']} ({len(series['points'])} pts)")

            ax.set_title(f'Plot {plot_idx} - Extracted Data (Calibrated)')
            ax.set_xlabel('X Value')
            ax.set_ylabel('Y Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

            st.pyplot(fig)
            plt.close(fig)

            # Show summary statistics
            total_points = sum(len(s['points']) for s in plot['series'])
            st.success(f"✅ Plot {plot_idx}: {len(plot['series'])} series, {total_points} points extracted")

if __name__ == "__main__":
    main()
