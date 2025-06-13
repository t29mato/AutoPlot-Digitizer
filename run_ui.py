#!/usr/bin/env python3
"""
Run the AutoPlot-Digitizer Streamlit UI
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit app"""
    ui_path = Path(__file__).parent / "ui" / "app.py"

    if not ui_path.exists():
        print(f"Error: UI app not found at {ui_path}")
        sys.exit(1)

    print("🚀 Starting AutoPlot-Digitizer Web UI...")
    print("📡 The app will open in your default browser")
    print("🛑 Press Ctrl+C to stop the server")

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(ui_path),
            "--server.headless", "false",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down AutoPlot-Digitizer UI...")
    except Exception as e:
        print(f"❌ Error starting UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
