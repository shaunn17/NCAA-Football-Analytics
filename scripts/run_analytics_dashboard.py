#!/usr/bin/env python3
"""
Script to run the Advanced Analytics Dashboard

This script launches the advanced analytics dashboard with all the new features:
- Trend analysis and forecasting
- Statistical analysis and metrics
- Comparative analytics
- Performance insights
- Predictive modeling
"""

import subprocess
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run the advanced analytics dashboard."""
    logger.info("🚀 Starting NCAA Football Advanced Analytics Dashboard...")
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Path to the analytics dashboard
    dashboard_path = project_root / "src" / "visualization" / "analytics_dashboard.py"
    
    if not dashboard_path.exists():
        logger.error(f"Analytics dashboard not found at {dashboard_path}")
        return 1
    
    try:
        # Run the analytics dashboard
        logger.info("📊 Launching Advanced Analytics Dashboard...")
        logger.info("🌐 Dashboard will be available at: http://localhost:8504")
        logger.info("📱 Use Ctrl+C to stop the dashboard")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", "8504",
            "--server.headless", "true"
        ], cwd=project_root)
        
    except KeyboardInterrupt:
        logger.info("🛑 Dashboard stopped by user")
    except Exception as e:
        logger.error(f"❌ Error running dashboard: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
