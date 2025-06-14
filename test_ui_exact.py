#!/usr/bin/env python3
"""
Simplified test to exactly reproduce the UI process_plot function behavior
"""
import sys
from pathlib import Path
import tempfile
import os

# Add src to path exactly like the UI does
project_root = Path(__file__).parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def create_temp_file_from_upload(file_path):
    """Simulate the temp file creation that UI does"""
    import shutil
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "temp_plot.png")
    shutil.copy2(file_path, temp_path)
    return temp_path

def cleanup_temp_files(tmp_path):
    """Clean up temporary files"""
    try:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            temp_dir = os.path.dirname(tmp_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
    except Exception as e:
        print(f"Warning: Could not clean up temp files: {e}")

def test_ui_process_plot():
    """Reproduce the exact process_plot function from UI"""
    print("🔍 Testing UI process_plot function behavior...")

    # Import exactly like the UI does
    try:
        from plot_digitizer.infrastructure.extractor_impl import OpenCVPlotExtractor
        print("✅ Import successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

    # Use the test image
    test_image_path = "test_data/s10853-012-6895-z_fig9.png"
    if not Path(test_image_path).exists():
        print(f"❌ Test image not found: {test_image_path}")
        return False

    tmp_path = None
    try:
        # Create temporary file (simulating UI upload)
        tmp_path = create_temp_file_from_upload(test_image_path)
        print(f"✅ Created temp file: {tmp_path}")

        # Process exactly like UI does
        extractor = OpenCVPlotExtractor(x_range=(0, 600), y_range=(0, 6))
        print("✅ Extractor created")

        # Execute extraction
        results = extractor.extract(tmp_path)
        print(f"✅ Extraction completed, found {len(results.plots)} plots")

        # Get detection results - this is where the error occurs
        try:
            detection_results = extractor.get_detection_results()
            print(f"✅ Detection results retrieved: {list(detection_results.keys())}")
        except AttributeError as e:
            print(f"❌ AttributeError getting detection results: {e}")
            print(f"❌ Available methods: {[m for m in dir(extractor) if not m.startswith('_')]}")
            return False

        # Create visualization images
        import cv2
        original_image = cv2.imread(tmp_path)
        if original_image is not None:
            try:
                detection_visualization = extractor.create_detection_visualization(original_image)
                detection_overview = extractor.create_detection_overview(original_image)
                print(f"✅ Visualizations created: {len(detection_visualization)} steps")
            except Exception as e:
                print(f"❌ Visualization error: {e}")
                return False
        else:
            print("❌ Could not load original image for visualization")
            return False

        print("✅ All UI functionality working correctly!")
        return True

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        cleanup_temp_files(tmp_path)

if __name__ == "__main__":
    success = test_ui_process_plot()
    if success:
        print("\n🎉 UI functionality test PASSED!")
    else:
        print("\n💥 UI functionality test FAILED!")
        sys.exit(1)
