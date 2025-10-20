#!/usr/bin/env python3
"""
Test script for College Football Data API connection and basic data collection
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from src.ingestion.api_client import create_api_client
from src.ingestion.data_collector import DataCollector
from config.settings import settings


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('api_test.log')
        ]
    )


def test_api_connection():
    """Test basic API connection"""
    print("🔍 Testing API connection...")
    
    client = create_api_client()
    
    if not client.api_key:
        print("❌ No API key found!")
        print("Please set COLLEGE_FOOTBALL_DATA_API_KEY in your environment or .env file")
        return False
    
    try:
        # Test connection with conferences endpoint
        conferences = client.get_conferences()
        print(f"✅ API connection successful! Found {len(conferences)} conferences")
        
        # Show first few conferences
        for conf in conferences[:5]:
            print(f"  - {conf.get('name', 'Unknown')} ({conf.get('short_name', 'N/A')})")
        
        return True
        
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False


def test_data_collection():
    """Test basic data collection"""
    print("\n📊 Testing data collection...")
    
    try:
        collector = DataCollector()
        
        # Test collecting conferences
        print("Collecting conferences...")
        conf_results = collector.collect_conferences()
        print(f"✅ Collected {conf_results['count']} conferences")
        
        # Test collecting teams
        print("Collecting teams...")
        teams_results = collector.collect_teams()
        print(f"✅ Collected {teams_results['count']} teams")
        
        # Test collecting one season of data
        print("Collecting team stats for 2023...")
        stats_results = collector.collect_team_stats(2023)
        print(f"✅ Collected stats for {stats_results['count']} teams")
        
        return True
        
    except Exception as e:
        print(f"❌ Data collection failed: {e}")
        return False


def main():
    """Main test function"""
    setup_logging()
    
    print("🏈 NCAA Football Analytics - API Test")
    print("=" * 50)
    
    # Test API connection
    if not test_api_connection():
        return 1
    
    # Test data collection
    if not test_data_collection():
        return 1
    
    print("\n🎉 All tests passed! API is ready for data collection.")
    print("\nNext steps:")
    print("1. Set up your database (PostgreSQL or DuckDB)")
    print("2. Run the full data collection pipeline")
    print("3. Start building visualizations and ML models")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


