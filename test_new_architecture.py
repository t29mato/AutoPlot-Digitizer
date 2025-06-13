#!/usr/bin/env python3
"""
Simple test script for the new plot-digitizer architecture
"""

import json
from pathlib import Path
from plot_digitizer.infrastructure.extractor_impl import OpenCVPlotExtractor
from plot_digitizer.use_cases.extract_plots import ExtractPlotsUseCase

def test_new_architecture():
    """Test the new Clean Architecture implementation"""

    # Test image path
    test_image = "test_data/s10853-012-6895-z_fig-9.png"
    output_file = "output/test_output_new.json"

    if not Path(test_image).exists():
        print(f"❌ Test image not found: {test_image}")
        return

    try:
        print("🧪 Testing new Clean Architecture implementation...")

        # Create extractor (infrastructure layer)
        extractor = OpenCVPlotExtractor()

        # Create use case (application layer)
        use_case = ExtractPlotsUseCase(extractor)

        # Execute extraction
        use_case.execute(test_image, output_file)

        # Read and display the result
        with open(output_file, 'r') as f:
            data = json.load(f)

        print("✅ Successfully extracted data!")
        print(f"📄 JSON output saved to: {output_file}")
        print(f"📊 Number of plots detected: {len(data['plots'])}")

        for plot in data['plots']:
            print(f"\n--- Plot {plot['id']} ---")
            print(f"  X-axis: {plot['x_axis']['label']} (unit: {plot['x_axis']['unit']})")
            print(f"  Y-axis: {plot['y_axis']['label']} (unit: {plot['y_axis']['unit']})")
            print(f"  Series count: {len(plot['series'])}")

            total_points = sum(len(s['points']) for s in plot['series'])
            print(f"  Total points: {total_points}")

            # Show sample of series
            for series in plot['series'][:3]:  # Show first 3 series
                points_count = len(series['points'])
                if points_count > 0:
                    sample_point = series['points'][0]
                    print(f"    - {series['sample']}: {points_count} points (e.g., {sample_point})")

        print(f"\n📝 Full JSON structure preview:")
        print(json.dumps(data, indent=2)[:500] + "..." if len(json.dumps(data)) > 500 else json.dumps(data, indent=2))

        # Create plot recreation if matplotlib is available
        try:
            recreate_plots_from_json(data, "output/recreated_plots")
        except ImportError:
            print("⚠️  matplotlib not available for plot recreation")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def recreate_plots_from_json(data, output_dir="output/recreated_plots"):
    """Recreate plots from extracted JSON data using matplotlib"""
    try:
        import matplotlib.pyplot as plt
        from pathlib import Path

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Color mapping for different series
        color_map = {
            'black': '#000000',
            'red1': '#FF0000',
            'red2': '#CC0000',
            'blue': '#0000FF',
            'teal': '#008080',
            'magenta': '#FF00FF'
        }

        for plot in data['plots']:
            plt.figure(figsize=(10, 8))

            plot_id = plot['id']
            x_axis = plot['x_axis']
            y_axis = plot['y_axis']

            # Set labels
            x_label = x_axis['label'] if x_axis['label'] else "X (pixels)"
            y_label = y_axis['label'] if y_axis['label'] else "Y (pixels)"

            if x_axis['unit']:
                x_label += f" ({x_axis['unit']})"
            if y_axis['unit']:
                y_label += f" ({y_axis['unit']})"

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(f"Recreated Plot {plot_id}")

            # Plot each series
            for series in plot['series']:
                if not series['points']:
                    continue

                x_coords = [p[0] for p in series['points']]
                y_coords = [p[1] for p in series['points']]

                # Get color for this series
                color = color_map.get(series['sample'], '#000000')

                # Plot points (note: y-coordinates are inverted for display)
                plt.scatter(x_coords, [-y for y in y_coords],
                           c=color, alpha=0.7, s=30, label=series['sample'])

            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save the plot
            output_path = Path(output_dir) / f"recreated_plot_{plot_id}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"📈 Recreated plot saved: {output_path}")

            plt.close()

    except ImportError:
        print("⚠️  matplotlib not available for plot recreation")
    except Exception as e:
        print(f"❌ Error recreating plots: {e}")

if __name__ == "__main__":
    test_new_architecture()
