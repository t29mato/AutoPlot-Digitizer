#!/usr/bin/env python3
"""
Test script to reproduce the UI error
"""
import sys
from pathlib import Path
import tempfile
import os

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_ui_functionality():
    """Test the functionality that's causing the error in the UI"""
    try:
        # Import the same way as in the UI
        from plot_digitizer.infrastructure.extractor_impl import OpenCVPlotExtractor

        print("✅ Import successful")

        # Create extractor the same way as in the UI
        extractor = OpenCVPlotExtractor(x_range=(0, 600), y_range=(0, 6))

        print("✅ Extractor creation successful")

        # Check if the method exists
        print(f"✅ Has get_detection_results: {hasattr(extractor, 'get_detection_results')}")
        print(f"✅ Has create_detection_visualization: {hasattr(extractor, 'create_detection_visualization')}")
        print(f"✅ Has create_detection_overview: {hasattr(extractor, 'create_detection_overview')}")

        # Test the method call without processing anything
        try:
            detection_results = extractor.get_detection_results()
            print(f"✅ get_detection_results() returned: {detection_results}")
        except Exception as e:
            print(f"❌ Error calling get_detection_results(): {e}")

        # Test with a dummy image
        test_image_path = "test_data/s10853-012-6895-z_fig9.png"
        if Path(test_image_path).exists():
            print(f"✅ Found test image: {test_image_path}")

            # Run extraction
            try:
                results = extractor.extract(test_image_path)
                print(f"✅ Extraction successful, found {len(results.plots)} plots")

                # Now test the detection results
                detection_results = extractor.get_detection_results()
                print(f"✅ Detection results keys: {list(detection_results.keys())}")

                # Test visualization creation
                import cv2
                original_image = cv2.imread(test_image_path)
                if original_image is not None:
                    step_viz = extractor.create_detection_visualization(original_image)
                    print(f"✅ Step visualizations created: {len(step_viz)}")

                    overview = extractor.create_detection_overview(original_image)
                    print(f"✅ Overview created with shape: {overview.shape}")
                else:
                    print("❌ Could not load test image")

            except Exception as e:
                print(f"❌ Error during extraction: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️  Test image not found: {test_image_path}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔍 Testing UI functionality...")
    test_ui_functionality()
    print("✅ Test completed!")
