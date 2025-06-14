"""
Simple test to verify the detection visualization functionality
"""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def test_detection_visualization():
    """Test the detection visualization with a sample image"""
    try:
        from plot_digitizer.infrastructure.extractor_impl import OpenCVPlotExtractor
        from plot_digitizer.infrastructure.detection_visualizer import DetectionVisualizer
        import cv2
        import numpy as np

        # Test with a sample image if available
        test_image_path = "test_data/s10853-012-6895-z_fig9.png"
        if not Path(test_image_path).exists():
            print("Test image not found, creating dummy test...")
            # Create a simple test image
            test_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
            cv2.rectangle(test_img, (50, 50), (550, 350), (0, 0, 0), 2)
            cv2.line(test_img, (100, 300), (500, 300), (0, 0, 0), 2)  # X-axis
            cv2.line(test_img, (100, 300), (100, 100), (0, 0, 0), 2)  # Y-axis
            # Add some sample data points
            cv2.circle(test_img, (150, 250), 3, (255, 0, 0), -1)
            cv2.circle(test_img, (200, 220), 3, (255, 0, 0), -1)
            cv2.circle(test_img, (250, 200), 3, (0, 255, 0), -1)
            cv2.circle(test_img, (300, 180), 3, (0, 255, 0), -1)

            cv2.imwrite("test_detection_image.png", test_img)
            test_image_path = "test_detection_image.png"

        print("Testing detection visualization...")

        # Create extractor
        extractor = OpenCVPlotExtractor(x_range=(0, 10), y_range=(0, 10))

        # Test detection
        results = extractor.extract(test_image_path)
        print(f"Extraction completed. Found {len(results.plots)} plots.")

        # Test visualization
        detection_results = extractor.get_detection_results()
        print(f"Detection results keys: {list(detection_results.keys())}")

        # Test visualization creation
        original_image = cv2.imread(test_image_path)
        if original_image is not None:
            step_visualizations = extractor.create_detection_visualization(original_image)
            print(f"Created {len(step_visualizations)} step visualizations")

            overview = extractor.create_detection_overview(original_image)
            print(f"Overview visualization shape: {overview.shape}")

            # Save test outputs
            for step_name, viz_img in step_visualizations.items():
                cv2.imwrite(f"test_output_{step_name}.png", viz_img)

            cv2.imwrite("test_output_overview.png", overview)
            print("Test visualizations saved!")

        print("✅ Detection visualization test completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_detection_visualization()
