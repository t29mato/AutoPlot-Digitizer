#!/usr/bin/env python3
"""
Multi-graph digitization example and test script
Updated for new Clean Architecture
"""

import json
from pathlib import Path

from plot_digitizer.domain.models import AxisInfo, PlotData, Series, DigitizationResult
from plot_digitizer.infrastructure.extractor_impl import OpenCVPlotExtractor
from plot_digitizer.use_cases.extract_plots import ExtractPlotsUseCase

def test_enhanced_digitization():
    """Test the enhanced digitization with new architecture."""
    image_path = "test_data/s10853-012-6895-z_fig-9.png"

    if not Path(image_path).exists():
        print(f"Error: Test image {image_path} not found!")
        return

    # Test with new architecture
    print("=== Testing Enhanced Multi-Graph Detection ===")

    try:
        # Create extractor and use case
        extractor = OpenCVPlotExtractor()
        use_case = ExtractPlotsUseCase(extractor)

        # Execute extraction - output to output folder
        output_file = "output/enhanced_multi_output.json"
        use_case.execute(image_path, output_file)

        # Read and analyze results
        with open(output_file, 'r') as f:
            data = json.load(f)

        result = DigitizationResult(**data)

        print(f"✓ Extracted data from {len(result.plots)} plots")

        # Display detailed analysis
        total_points = 0
        for plot in result.plots:
            plot_points = sum(len(series.points) for series in plot.series)
            total_points += plot_points

            print(f"\nPlot {plot.id}:")
            print(f"  X-axis: {plot.x_axis.label} ({plot.x_axis.unit})")
            print(f"  Y-axis: {plot.y_axis.label} ({plot.y_axis.unit})")
            print(f"  Series: {len(plot.series)}")
            print(f"  Points: {plot_points}")

            # Show series details
            for series in plot.series:
                if len(series.points) > 0:
                    x_coords = [p[0] for p in series.points]
                    y_coords = [p[1] for p in series.points]
                    print(f"    {series.sample}: {len(series.points)} points")
                    print(f"      X range: {min(x_coords):.1f} - {max(x_coords):.1f}")
                    print(f"      Y range: {min(y_coords):.1f} - {max(y_coords):.1f}")

        print(f"\nTotal data points extracted: {total_points}")

        # Create summary JSON
        summary = {
            "total_plots": len(result.plots),
            "total_points": total_points,
            "plots_summary": [
                {
                    "id": plot.id,
                    "x_axis": {"label": plot.x_axis.label, "unit": plot.x_axis.unit},
                    "y_axis": {"label": plot.y_axis.label, "unit": plot.y_axis.unit},
                    "series_count": len(plot.series),
                    "total_points": sum(len(series.points) for series in plot.series),
                    "series_details": [
                        {
                            "name": series.sample,
                            "points_count": len(series.points)
                        } for series in plot.series
                    ]
                } for plot in result.plots
            ]
        }

        summary_file = "output/extraction_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nSummary saved to: {summary_file}")

    except Exception as e:
        print(f"Error during digitization: {e}")
        import traceback
        traceback.print_exc()

def demo_multi_graph_workflow():
    """Demonstrate the complete multi-graph workflow."""
    print("\n=== Multi-Graph Workflow Demo ===")

    workflow_steps = [
        "1. Load image with multiple plots",
        "2. Detect individual plot regions automatically",
        "3. Extract axis labels using OCR",
        "4. Identify marker colors and positions",
        "5. Cluster markers into data points",
        "6. Export structured JSON with all plot data"
    ]

    print("Workflow steps:")
    for step in workflow_steps:
        print(f"  {step}")

    print("\nJSON Output Structure:")
    example_structure = {
        "plots": [
            {
                "id": 0,
                "x_axis": {"label": "Temperature", "unit": "K"},
                "y_axis": {"label": "Thermal Conductivity", "unit": "W/mK"},
                "series": [
                    {
                        "sample": "red1",
                        "points": [[300.5, 2.1], [350.2, 1.8]]
                    }
                ]
            }
        ]
    }

    print(json.dumps(example_structure, indent=2))

if __name__ == "__main__":
    test_enhanced_digitization()
    demo_multi_graph_workflow()
