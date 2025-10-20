#!/usr/bin/env python3
"""
Complete Data Pipeline for NCAA Football Analytics

This script runs the complete data pipeline from API ingestion to ML-ready datasets.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import argparse
from datetime import datetime
import pandas as pd

from src.ingestion.data_collector import DataCollector
from src.processing.cleaner import DataCleaner
from src.processing.transformer import DataTransformer
from config.settings import settings


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def run_data_pipeline(seasons: list = None, 
                     conferences: list = None,
                     api_key: str = None,
                     skip_collection: bool = False,
                     skip_cleaning: bool = False,
                     skip_transformation: bool = False) -> dict:
    """
    Run the complete data pipeline
    
    Args:
        seasons: List of seasons to process
        conferences: List of conferences to focus on
        api_key: API key for data collection
        skip_collection: Skip data collection step
        skip_cleaning: Skip data cleaning step
        skip_transformation: Skip data transformation step
        
    Returns:
        Dictionary with pipeline results
    """
    logger = logging.getLogger(__name__)
    
    pipeline_results = {
        "start_time": datetime.now().isoformat(),
        "seasons": seasons or settings.seasons_to_collect,
        "conferences": conferences or settings.conferences_to_focus,
        "steps": {}
    }
    
    try:
        # Step 1: Data Collection
        if not skip_collection:
            logger.info("=" * 60)
            logger.info("STEP 1: DATA COLLECTION")
            logger.info("=" * 60)
            
            collector = DataCollector(api_key)
            collection_results = collector.collect_all_data(seasons, conferences)
            
            pipeline_results["steps"]["collection"] = collection_results
            
            if not collection_results.get("success", False):
                raise Exception(f"Data collection failed: {collection_results.get('error', 'Unknown error')}")
            
            logger.info("✅ Data collection completed successfully")
        else:
            logger.info("⏭️  Skipping data collection")
            pipeline_results["steps"]["collection"] = {"skipped": True}
        
        # Step 2: Data Cleaning
        if not skip_cleaning:
            logger.info("=" * 60)
            logger.info("STEP 2: DATA CLEANING")
            logger.info("=" * 60)
            
            cleaner = DataCleaner()
            cleaning_results = cleaner.process_all_data(seasons)
            
            pipeline_results["steps"]["cleaning"] = cleaning_results
            
            if not cleaning_results.get("success", False):
                raise Exception(f"Data cleaning failed: {cleaning_results.get('error', 'Unknown error')}")
            
            logger.info("✅ Data cleaning completed successfully")
        else:
            logger.info("⏭️  Skipping data cleaning")
            pipeline_results["steps"]["cleaning"] = {"skipped": True}
        
        # Step 3: Data Transformation
        if not skip_transformation:
            logger.info("=" * 60)
            logger.info("STEP 3: DATA TRANSFORMATION")
            logger.info("=" * 60)
            
            transformer = DataTransformer()
            
            # Load cleaned data
            processed_data_dir = settings.processed_data_dir
            
            # Combine team stats across seasons
            team_stats_files = list(processed_data_dir.glob("team_stats_*_clean.csv"))
            if not team_stats_files:
                raise Exception("No cleaned team stats files found")
            
            logger.info(f"Found {len(team_stats_files)} team stats files")
            
            all_team_stats = []
            for file_path in team_stats_files:
                df = pd.read_csv(file_path)
                all_team_stats.append(df)
                logger.info(f"Loaded {len(df)} records from {file_path.name}")
            
            combined_team_stats = pd.concat(all_team_stats, ignore_index=True)
            logger.info(f"Combined team stats: {len(combined_team_stats)} total records")
            
            # Create features
            logger.info("Creating team features...")
            features_df = transformer.create_team_features(combined_team_stats)
            
            # Create prediction dataset
            logger.info("Creating prediction dataset...")
            ml_df = transformer.create_prediction_dataset(features_df)
            
            # Export ML dataset
            ml_file = transformer.export_for_ml(ml_df, "ncaa_football_ml_dataset.csv")
            
            transformation_results = {
                "success": True,
                "total_records": len(combined_team_stats),
                "feature_records": len(features_df),
                "ml_records": len(ml_df),
                "ml_file": ml_file,
                "features_count": len(ml_df.columns) - 5  # Exclude identifier columns
            }
            
            pipeline_results["steps"]["transformation"] = transformation_results
            
            logger.info("✅ Data transformation completed successfully")
            logger.info(f"📊 Final ML dataset: {len(ml_df)} records, {transformation_results['features_count']} features")
            logger.info(f"💾 Saved to: {ml_file}")
        else:
            logger.info("⏭️  Skipping data transformation")
            pipeline_results["steps"]["transformation"] = {"skipped": True}
        
        pipeline_results["end_time"] = datetime.now().isoformat()
        pipeline_results["success"] = True
        
        logger.info("=" * 60)
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
        return pipeline_results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        pipeline_results["error"] = str(e)
        pipeline_results["success"] = False
        pipeline_results["end_time"] = datetime.now().isoformat()
        return pipeline_results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run NCAA Football Analytics Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with default settings
  python scripts/run_pipeline.py
  
  # Run pipeline for specific seasons
  python scripts/run_pipeline.py --seasons 2022 2023 2024
  
  # Run pipeline with custom API key
  python scripts/run_pipeline.py --api-key your_api_key_here
  
  # Skip data collection (use existing data)
  python scripts/run_pipeline.py --skip-collection
  
  # Run only transformation (skip collection and cleaning)
  python scripts/run_pipeline.py --skip-collection --skip-cleaning
        """
    )
    
    parser.add_argument("--seasons", nargs="+", type=int,
                       help="Seasons to process (default: from settings)")
    parser.add_argument("--conferences", nargs="+",
                       help="Conferences to focus on (default: from settings)")
    parser.add_argument("--api-key", help="College Football Data API key")
    parser.add_argument("--skip-collection", action="store_true",
                       help="Skip data collection step")
    parser.add_argument("--skip-cleaning", action="store_true",
                       help="Skip data cleaning step")
    parser.add_argument("--skip-transformation", action="store_true",
                       help="Skip data transformation step")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Print pipeline info
    logger.info("🏈 NCAA Football Analytics - Data Pipeline")
    logger.info("=" * 60)
    logger.info(f"Seasons: {args.seasons or settings.seasons_to_collect}")
    logger.info(f"Conferences: {args.conferences or settings.conferences_to_focus}")
    logger.info(f"Skip collection: {args.skip_collection}")
    logger.info(f"Skip cleaning: {args.skip_cleaning}")
    logger.info(f"Skip transformation: {args.skip_transformation}")
    logger.info("=" * 60)
    
    # Run pipeline
    results = run_data_pipeline(
        seasons=args.seasons,
        conferences=args.conferences,
        api_key=args.api_key,
        skip_collection=args.skip_collection,
        skip_cleaning=args.skip_cleaning,
        skip_transformation=args.skip_transformation
    )
    
    # Print results
    if results["success"]:
        logger.info("✅ Pipeline completed successfully!")
        
        # Print summary
        for step_name, step_results in results["steps"].items():
            if step_results.get("skipped"):
                logger.info(f"  {step_name}: Skipped")
            elif step_results.get("success"):
                logger.info(f"  {step_name}: ✅ Success")
            else:
                logger.info(f"  {step_name}: ❌ Failed")
        
        # Print final dataset info if transformation was run
        if "transformation" in results["steps"] and not results["steps"]["transformation"].get("skipped"):
            tf_results = results["steps"]["transformation"]
            logger.info(f"📊 Final dataset: {tf_results['ml_records']} records, {tf_results['features_count']} features")
            logger.info(f"💾 Saved to: {tf_results['ml_file']}")
        
        return 0
    else:
        logger.error("❌ Pipeline failed!")
        logger.error(f"Error: {results.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


