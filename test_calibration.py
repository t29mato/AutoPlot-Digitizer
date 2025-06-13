#!/usr/bin/env python3
"""
Test script to run plot digitizer with custom calibration settings
Updated for new Clean Architecture
"""

import pandas as pd
import os
import json
from pathlib import Path

from plot_digitizer.domain.models import DigitizationResult
from plot_digitizer.infrastructure.extractor_impl import OpenCVPlotExtractor
from plot_digitizer.use_cases.extract_plots import ExtractPlotsUseCase

def test_with_different_ranges():
    """Test digitization with different value ranges using new architecture"""
    image_path = "test_data/s10853-012-6895-z_fig-9.png"

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    test_configs = [
        {
            "name": "default",
            "output": "output/test_output_default.json"
        },
        {
            "name": "narrow_analysis",
            "output": "output/test_output_narrow.json"
        },
        {
            "name": "temperature_analysis",
            "output": "output/test_output_temp.json"
        }
    ]

    extractor = OpenCVPlotExtractor()
    use_case = ExtractPlotsUseCase(extractor)

    for config in test_configs:
        print(f"\n=== Testing {config['name']} configuration ===")

        try:
            # Execute extraction
            use_case.execute(image_path, config['output'])

            # Read and analyze results
            with open(config['output'], 'r') as f:
                data = json.load(f)

            result = DigitizationResult(**data)

            # Analyze results
            total_points = sum(len(series.points) for plot in result.plots for series in plot.series)
            print(f"Extracted {total_points} total points from {len(result.plots)} plots")

            # Show detailed analysis for each plot
            for plot in result.plots:
                plot_points = sum(len(series.points) for series in plot.series)
                print(f"\nPlot {plot.id}:")
                print(f"  X-axis: {plot.x_axis.label} ({plot.x_axis.unit})")
                print(f"  Y-axis: {plot.y_axis.label} ({plot.y_axis.unit})")
                print(f"  Series: {len(plot.series)}")
                print(f"  Points: {plot_points}")

                # Show sample data from first few series
                for series in plot.series[:3]:  # Show first 3 series
                    if len(series.points) > 0:
                        x_coords = [p[0] for p in series.points]
                        y_coords = [p[1] for p in series.points]
                        print(f"    {series.sample}: {len(series.points)} points")
                        print(f"      X range: {min(x_coords):.2f} to {max(x_coords):.2f}")
                        print(f"      Y range: {min(y_coords):.2f} to {max(y_coords):.2f}")

            # Also create CSV output for backward compatibility
            csv_output = config['output'].replace('.json', '.csv')
            create_csv_from_json(config['output'], csv_output)
            print(f"CSV output saved to: {csv_output}")

        except Exception as e:
            print(f"Error with {config['name']}: {e}")
            import traceback
            traceback.print_exc()


def create_csv_from_json(json_file, csv_file):
    """Convert JSON output to CSV format for compatibility"""
    with open(json_file, 'r') as f:
        data = json.load(f)

    result = DigitizationResult(**data)

    # Create CSV data
    csv_data = []
    for plot in result.plots:
        for series in plot.series:
            for point in series.points:
                csv_data.append({
                    'Plot_ID': plot.id,
                    'Series': series.sample,
                    'X': point[0],
                    'Y': point[1],
                    'X_Label': plot.x_axis.label,
                    'X_Unit': plot.x_axis.unit,
                    'Y_Label': plot.y_axis.label,
                    'Y_Unit': plot.y_axis.unit
                })

    # Save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    return df

if __name__ == "__main__":
    test_with_different_ranges()
