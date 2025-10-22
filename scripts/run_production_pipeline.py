#!/usr/bin/env python3
"""
Production Pipeline Runner Scripts

This script provides various ways to run the production pipeline:
- Run once: python scripts/run_production_pipeline.py run
- Start scheduler: python scripts/run_production_pipeline.py schedule
- Monitor: python scripts/run_production_pipeline.py monitor
"""

import sys
import subprocess
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_pipeline_once():
    """Run the production pipeline once."""
    logger.info("🚀 Running production pipeline once...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "src.pipeline.production_pipeline", "run"
        ], cwd=Path(__file__).parent.parent)
        
        if result.returncode == 0:
            logger.info("✅ Pipeline completed successfully")
        else:
            logger.error("❌ Pipeline failed")
        
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"❌ Error running pipeline: {e}")
        return False

def start_scheduler():
    """Start the production pipeline scheduler."""
    logger.info("⏰ Starting production pipeline scheduler...")
    logger.info("📊 Pipeline will run on schedule defined in config/pipeline_config.yaml")
    logger.info("🛑 Use Ctrl+C to stop the scheduler")
    
    try:
        subprocess.run([
            sys.executable, "-m", "src.pipeline.production_pipeline"
        ], cwd=Path(__file__).parent.parent)
        
    except KeyboardInterrupt:
        logger.info("🛑 Scheduler stopped by user")
    except Exception as e:
        logger.error(f"❌ Error starting scheduler: {e}")

def start_monitor():
    """Start the pipeline monitoring dashboard."""
    logger.info("📊 Starting pipeline monitoring dashboard...")
    logger.info("🌐 Dashboard will be available at: http://localhost:8505")
    logger.info("🛑 Use Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "src/visualization/pipeline_monitor.py",
            "--server.port", "8505",
            "--server.headless", "true"
        ], cwd=Path(__file__).parent.parent)
        
    except KeyboardInterrupt:
        logger.info("🛑 Monitor stopped by user")
    except Exception as e:
        logger.error(f"❌ Error starting monitor: {e}")

def show_help():
    """Show help information."""
    print("""
🏭 Production Pipeline Runner

Usage:
    python scripts/run_production_pipeline.py [command]

Commands:
    run        Run the pipeline once
    schedule   Start the pipeline scheduler (runs on schedule)
    monitor    Start the monitoring dashboard
    help       Show this help message

Examples:
    # Run pipeline once
    python scripts/run_production_pipeline.py run
    
    # Start scheduler (runs daily at 6 AM)
    python scripts/run_production_pipeline.py schedule
    
    # Start monitoring dashboard
    python scripts/run_production_pipeline.py monitor

Configuration:
    Edit config/pipeline_config.yaml to configure:
    - Schedule timing
    - Data collection settings
    - Notification settings
    - Performance settings
""")

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    if command == "run":
        success = run_pipeline_once()
        sys.exit(0 if success else 1)
    elif command == "schedule":
        start_scheduler()
    elif command == "monitor":
        start_monitor()
    elif command == "help":
        show_help()
    else:
        print(f"❌ Unknown command: {command}")
        show_help()
        sys.exit(1)

if __name__ == "__main__":
    main()


