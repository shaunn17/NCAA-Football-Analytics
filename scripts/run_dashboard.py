#!/usr/bin/env python3
"""
Dashboard Runner Script

This script provides easy access to both Dash and Streamlit dashboard versions.
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_dash_dashboard():
    """Run the Plotly Dash dashboard"""
    print("🚀 Starting Plotly Dash Dashboard...")
    print("📊 Dashboard will be available at: http://127.0.0.1:8050")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, 
            "src/visualization/dashboard.py"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dashboard: {e}")

def run_streamlit_dashboard():
    """Run the Streamlit dashboard"""
    print("🚀 Starting Streamlit Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    try:
        subprocess.run([
            "streamlit", "run", 
            "src/visualization/streamlit_dashboard.py",
            "--server.port", "8501",
            "--server.address", "127.0.0.1"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dashboard: {e}")

def run_enhanced_dashboard():
    """Run the enhanced Streamlit dashboard with ML predictions"""
    print("🚀 Starting Enhanced Streamlit Dashboard with ML Predictions...")
    print("📊 Dashboard will be available at: http://localhost:8502")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    try:
        subprocess.run([
            "streamlit", "run", 
            "src/visualization/enhanced_dashboard.py",
            "--server.port", "8502",
            "--server.address", "127.0.0.1"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dashboard: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Run NCAA Football Analytics Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run enhanced dashboard with ML predictions (recommended)
  python scripts/run_dashboard.py enhanced
  
  # Run basic Streamlit dashboard
  python scripts/run_dashboard.py streamlit
  
  # Run Plotly Dash dashboard
  python scripts/run_dashboard.py dash
        """
    )
    
    parser.add_argument(
        "dashboard_type",
        choices=["dash", "streamlit", "enhanced"],
        help="Type of dashboard to run"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        help="Port number for the dashboard"
    )
    
    args = parser.parse_args()
    
    # Check if data exists
    data_path = Path("data/models/ncaa_football_ml_dataset.csv")
    if not data_path.exists():
        print("❌ Error: No data found!")
        print("Please run the data pipeline first:")
        print("  python scripts/run_pipeline.py")
        return 1
    
    print("🏈 NCAA Football Analytics Dashboard")
    print("=" * 50)
    
    if args.dashboard_type == "dash":
        run_dash_dashboard()
    elif args.dashboard_type == "streamlit":
        run_streamlit_dashboard()
    elif args.dashboard_type == "enhanced":
        run_enhanced_dashboard()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
