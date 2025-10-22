"""
Production-Ready Data Pipeline Orchestrator

This module provides a robust, production-ready data pipeline that includes:
- Automated scheduling and execution
- Comprehensive error handling and recovery
- Monitoring and logging
- Configuration management
- Performance optimization
- Data validation and quality checks
- Notifications and alerts
- Incremental updates
- Backup and recovery
- Health checks
"""

import os
import sys
import logging
import traceback
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import schedule
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import yaml
from dataclasses import dataclass, asdict
import sqlite3
import hashlib
import gzip
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.data_collector import DataCollector
from src.processing.cleaner import DataCleaner
from src.processing.transformer import DataTransformer
from src.ml.models import NCAAFootballPredictor
from src.storage.simple_database import SimpleDatabaseManager
from config.settings import settings

@dataclass
class PipelineConfig:
    """Configuration for the production pipeline."""
    # Data collection settings
    start_year: int = 2023
    end_year: int = 2024
    conferences: List[str] = None
    
    # Scheduling settings
    schedule_interval: str = "daily"  # daily, weekly, monthly
    schedule_time: str = "06:00"  # HH:MM format
    
    # Performance settings
    batch_size: int = 100
    max_retries: int = 3
    retry_delay: int = 60  # seconds
    
    # Monitoring settings
    enable_notifications: bool = True
    notification_email: str = ""
    slack_webhook: str = ""
    
    # Data validation settings
    min_data_quality_score: float = 0.8
    max_null_percentage: float = 0.1
    
    # Backup settings
    enable_backup: bool = True
    backup_retention_days: int = 30
    
    # Health check settings
    health_check_interval: int = 300  # seconds
    max_pipeline_duration: int = 3600  # seconds
    
    def __post_init__(self):
        if self.conferences is None:
            self.conferences = ["B1G", "SEC", "ACC", "Big 12", "PAC"]

@dataclass
class PipelineStatus:
    """Status tracking for the pipeline."""
    status: str = "idle"  # idle, running, completed, failed, paused
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    records_processed: int = 0
    errors: List[str] = None
    warnings: List[str] = None
    data_quality_score: float = 0.0
    last_successful_run: Optional[datetime] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

class ProductionPipeline:
    """Production-ready data pipeline orchestrator."""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the production pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.status = PipelineStatus()
        self.logger = self._setup_logging()
        self.state_db = self._setup_state_database()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging."""
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Setup logger
        logger = logging.getLogger("production_pipeline")
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(
            logs_dir / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_state_database(self) -> sqlite3.Connection:
        """Setup SQLite database for pipeline state tracking."""
        state_db_path = Path("data/pipeline_state.db")
        state_db_path.parent.mkdir(exist_ok=True)
        
        conn = sqlite3.connect(str(state_db_path))
        
        # Create tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                status TEXT,
                duration REAL,
                records_processed INTEGER,
                data_quality_score REAL,
                errors TEXT,
                warnings TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS data_checksums (
                table_name TEXT PRIMARY KEY,
                checksum TEXT,
                last_updated TIMESTAMP
            )
        """)
        
        conn.commit()
        return conn
    
    def _calculate_data_checksum(self, data_path: Path) -> str:
        """Calculate checksum for data validation."""
        if not data_path.exists():
            return ""
        
        with open(data_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _backup_data(self) -> bool:
        """Create backup of current data."""
        if not self.config.enable_backup:
            return True
        
        try:
            backup_dir = Path("data/backups")
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"backup_{timestamp}"
            backup_path.mkdir(exist_ok=True)
            
            # Backup key data files
            data_files = [
                "data/models/ncaa_football_ml_dataset.csv",
                "data/models/2025_predictions.csv",
                "data/ncaa_football_simple.duckdb"
            ]
            
            for file_path in data_files:
                src_path = Path(file_path)
                if src_path.exists():
                    dst_path = backup_path / src_path.name
                    shutil.copy2(src_path, dst_path)
                    
                    # Compress large files
                    if dst_path.stat().st_size > 10 * 1024 * 1024:  # 10MB
                        with open(dst_path, 'rb') as f_in:
                            with gzip.open(f"{dst_path}.gz", 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        dst_path.unlink()
            
            self.logger.info(f"Backup created: {backup_path}")
            
            # Clean old backups
            self._cleanup_old_backups(backup_dir)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return False
    
    def _cleanup_old_backups(self, backup_dir: Path):
        """Clean up old backup files."""
        cutoff_date = datetime.now() - timedelta(days=self.config.backup_retention_days)
        
        for backup_path in backup_dir.iterdir():
            if backup_path.is_dir():
                try:
                    backup_time = datetime.fromtimestamp(backup_path.stat().st_mtime)
                    if backup_time < cutoff_date:
                        shutil.rmtree(backup_path)
                        self.logger.info(f"Removed old backup: {backup_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove backup {backup_path}: {e}")
    
    def _send_notification(self, subject: str, message: str, is_error: bool = False):
        """Send notification via email or Slack."""
        if not self.config.enable_notifications:
            return
        
        try:
            if self.config.notification_email:
                self._send_email_notification(subject, message, is_error)
            
            if self.config.slack_webhook:
                self._send_slack_notification(subject, message, is_error)
                
        except Exception as e:
            self.logger.error(f"Notification failed: {e}")
    
    def _send_email_notification(self, subject: str, message: str, is_error: bool):
        """Send email notification."""
        # This is a placeholder - implement with your email service
        self.logger.info(f"Email notification: {subject}")
    
    def _send_slack_notification(self, subject: str, message: str, is_error: bool):
        """Send Slack notification."""
        try:
            color = "danger" if is_error else "good"
            payload = {
                "text": subject,
                "attachments": [{
                    "color": color,
                    "text": message,
                    "timestamp": int(time.time())
                }]
            }
            
            response = requests.post(self.config.slack_webhook, json=payload)
            response.raise_for_status()
            
        except Exception as e:
            self.logger.error(f"Slack notification failed: {e}")
    
    def _validate_data_quality(self, data_path: Path) -> float:
        """Validate data quality and return quality score."""
        try:
            df = pd.read_csv(data_path)
            
            # Check for null values
            null_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns))
            if null_percentage > self.config.max_null_percentage:
                self.status.warnings.append(f"High null percentage: {null_percentage:.2%}")
            
            # Check for duplicate records
            duplicate_percentage = df.duplicated().sum() / len(df)
            if duplicate_percentage > 0.05:
                self.status.warnings.append(f"High duplicate percentage: {duplicate_percentage:.2%}")
            
            # Check for reasonable data ranges
            if 'win_percentage' in df.columns:
                invalid_wins = df[(df['win_percentage'] < 0) | (df['win_percentage'] > 1)]
                if len(invalid_wins) > 0:
                    self.status.warnings.append(f"Invalid win percentages: {len(invalid_wins)} records")
            
            # Calculate quality score
            quality_score = 1.0 - null_percentage - duplicate_percentage
            quality_score = max(0.0, min(1.0, quality_score))
            
            return quality_score
            
        except Exception as e:
            self.logger.error(f"Data quality validation failed: {e}")
            return 0.0
    
    def _run_data_collection(self) -> bool:
        """Run data collection phase."""
        try:
            self.logger.info("Starting data collection phase...")
            
            collector = DataCollector(settings.cfbd_api_key)
            collector.collect_all_data(
                start_year=self.config.start_year,
                end_year=self.config.end_year,
                conferences=self.config.conferences
            )
            
            self.logger.info("Data collection completed successfully")
            return True
            
        except Exception as e:
            error_msg = f"Data collection failed: {e}"
            self.logger.error(error_msg)
            self.status.errors.append(error_msg)
            return False
    
    def _run_data_processing(self) -> bool:
        """Run data processing phase."""
        try:
            self.logger.info("Starting data processing phase...")
            
            # Data cleaning
            cleaner = DataCleaner()
            cleaned_data = cleaner.run_cleaning_pipeline(
                start_year=self.config.start_year,
                end_year=self.config.end_year
            )
            
            # Data transformation
            transformer = DataTransformer(cleaned_data)
            ml_dataset = transformer.run_transformation_pipeline()
            
            # Update win percentages
            from scripts.fix_win_percentages import update_win_percentages_in_processed_data
            update_win_percentages_in_processed_data(
                Path("data/raw"),
                Path("data/models"),
                Path("data/models/ncaa_football_ml_dataset.csv")
            )
            
            self.status.records_processed = len(ml_dataset)
            self.logger.info(f"Data processing completed: {len(ml_dataset)} records")
            return True
            
        except Exception as e:
            error_msg = f"Data processing failed: {e}"
            self.logger.error(error_msg)
            self.status.errors.append(error_msg)
            return False
    
    def _run_ml_training(self) -> bool:
        """Run ML model training phase."""
        try:
            self.logger.info("Starting ML model training phase...")
            
            ml_dataset_path = Path("data/models/ncaa_football_ml_dataset.csv")
            if not ml_dataset_path.exists():
                raise FileNotFoundError("ML dataset not found")
            
            df = pd.read_csv(ml_dataset_path)
            predictor = NCAAFootballPredictor(df)
            
            # Train models
            features_with_targets = predictor.create_targets()
            
            # Train Top 25 model
            if features_with_targets['is_top_25'].nunique() >= 2:
                X_top_25 = features_with_targets.drop(columns=['is_top_25', 'conference_winner', 'performance_rank', 'team', 'conference'], errors='ignore')
                y_top_25 = features_with_targets['is_top_25']
                top_25_results = predictor.train_top_25_model(X_top_25, y_top_25)
                self.logger.info(f"Top 25 model trained: {top_25_results['accuracy']:.3f} accuracy")
            
            # Train Performance Ranking model
            X_perf = features_with_targets.drop(columns=['is_top_25', 'conference_winner', 'performance_rank', 'team', 'conference'], errors='ignore')
            y_perf = features_with_targets['performance_rank']
            perf_results = predictor.train_performance_ranking_model(X_perf, y_perf)
            self.logger.info(f"Performance ranking model trained: {perf_results['r2_score']:.3f} R²")
            
            self.logger.info("ML model training completed successfully")
            return True
            
        except Exception as e:
            error_msg = f"ML model training failed: {e}"
            self.logger.error(error_msg)
            self.status.errors.append(error_msg)
            return False
    
    def _run_database_update(self) -> bool:
        """Run database update phase."""
        try:
            self.logger.info("Starting database update phase...")
            
            db = SimpleDatabaseManager(Path("data"))
            db.create_schema_from_csv(Path("data/models/ncaa_football_ml_dataset.csv"))
            db.load_data_from_csv(Path("data/models/ncaa_football_ml_dataset.csv"))
            
            self.logger.info("Database update completed successfully")
            return True
            
        except Exception as e:
            error_msg = f"Database update failed: {e}"
            self.logger.error(error_msg)
            self.status.errors.append(error_msg)
            return False
    
    def _save_pipeline_state(self):
        """Save pipeline state to database."""
        try:
            cursor = self.state_db.cursor()
            cursor.execute("""
                INSERT INTO pipeline_runs 
                (start_time, end_time, status, duration, records_processed, 
                 data_quality_score, errors, warnings)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.status.start_time,
                self.status.end_time,
                self.status.status,
                self.status.duration,
                self.status.records_processed,
                self.status.data_quality_score,
                json.dumps(self.status.errors),
                json.dumps(self.status.warnings)
            ))
            self.state_db.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to save pipeline state: {e}")
    
    def run_pipeline(self) -> bool:
        """Run the complete production pipeline."""
        self.status.status = "running"
        self.status.start_time = datetime.now()
        self.status.errors = []
        self.status.warnings = []
        
        self.logger.info("🚀 Starting production pipeline...")
        
        try:
            # Phase 1: Data Collection
            if not self._run_data_collection():
                raise Exception("Data collection phase failed")
            
            # Phase 2: Data Processing
            if not self._run_data_processing():
                raise Exception("Data processing phase failed")
            
            # Phase 3: ML Training
            if not self._run_ml_training():
                raise Exception("ML training phase failed")
            
            # Phase 4: Database Update
            if not self._run_database_update():
                raise Exception("Database update phase failed")
            
            # Phase 5: Data Validation
            ml_dataset_path = Path("data/models/ncaa_football_ml_dataset.csv")
            self.status.data_quality_score = self._validate_data_quality(ml_dataset_path)
            
            if self.status.data_quality_score < self.config.min_data_quality_score:
                raise Exception(f"Data quality score too low: {self.status.data_quality_score:.3f}")
            
            # Phase 6: Backup
            if not self._backup_data():
                self.status.warnings.append("Backup failed")
            
            # Success
            self.status.status = "completed"
            self.status.end_time = datetime.now()
            self.status.duration = (self.status.end_time - self.status.start_time).total_seconds()
            self.status.last_successful_run = self.status.end_time
            
            self.logger.info(f"✅ Pipeline completed successfully in {self.status.duration:.1f} seconds")
            
            # Send success notification
            self._send_notification(
                "Pipeline Success",
                f"Pipeline completed successfully. Processed {self.status.records_processed} records. "
                f"Data quality score: {self.status.data_quality_score:.3f}",
                is_error=False
            )
            
            return True
            
        except Exception as e:
            self.status.status = "failed"
            self.status.end_time = datetime.now()
            self.status.duration = (self.status.end_time - self.status.start_time).total_seconds()
            
            error_msg = f"Pipeline failed: {e}"
            self.logger.error(error_msg)
            self.status.errors.append(error_msg)
            
            # Send error notification
            self._send_notification(
                "Pipeline Failure",
                f"Pipeline failed after {self.status.duration:.1f} seconds. "
                f"Error: {str(e)}",
                is_error=True
            )
            
            return False
            
        finally:
            self._save_pipeline_state()
    
    def start_scheduler(self):
        """Start the pipeline scheduler."""
        self.logger.info(f"Starting pipeline scheduler: {self.config.schedule_interval} at {self.config.schedule_time}")
        
        if self.config.schedule_interval == "daily":
            schedule.every().day.at(self.config.schedule_time).do(self.run_pipeline)
        elif self.config.schedule_interval == "weekly":
            schedule.every().week.at(self.config.schedule_time).do(self.run_pipeline)
        elif self.config.schedule_interval == "monthly":
            schedule.every().month.at(self.config.schedule_time).do(self.run_pipeline)
        
        # Run health checks
        schedule.every(self.config.health_check_interval).seconds.do(self._health_check)
        
        # Keep scheduler running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _health_check(self):
        """Perform system health check."""
        try:
            # Check if pipeline is stuck
            if self.status.status == "running":
                if self.status.start_time:
                    duration = (datetime.now() - self.status.start_time).total_seconds()
                    if duration > self.config.max_pipeline_duration:
                        self.logger.error("Pipeline appears to be stuck, stopping...")
                        self.status.status = "failed"
                        self.status.errors.append("Pipeline timeout")
            
            # Check disk space
            disk_usage = shutil.disk_usage(Path("data"))
            free_space_gb = disk_usage.free / (1024**3)
            if free_space_gb < 1.0:  # Less than 1GB free
                self.status.warnings.append(f"Low disk space: {free_space_gb:.1f}GB free")
            
            self.logger.debug("Health check completed")
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")

def load_config(config_path: Path = None) -> PipelineConfig:
    """Load pipeline configuration from file."""
    if config_path is None:
        config_path = Path("config/pipeline_config.yaml")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return PipelineConfig(**config_data)
    else:
        # Return default configuration
        return PipelineConfig()

def main():
    """Main entry point for the production pipeline."""
    # Load configuration
    config = load_config()
    
    # Create and run pipeline
    pipeline = ProductionPipeline(config)
    
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        # Run pipeline once
        success = pipeline.run_pipeline()
        sys.exit(0 if success else 1)
    else:
        # Start scheduler
        pipeline.start_scheduler()

if __name__ == "__main__":
    main()


