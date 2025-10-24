#!/usr/bin/env python3
"""
Quick API Usage Checker

Simple script to check your current API usage and rate limit status.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.api_client import create_api_client


def main():
    """Check current API usage"""
    print("🏈 College Football Data API Usage Check")
    print("=" * 40)
    
    try:
        # Create API client
        client = create_api_client()
        
        # Get usage stats
        stats = client.get_usage_stats()
        
        print(f"📊 Current Usage:")
        print(f"  Total Requests: {stats['total_requests']}")
        print(f"  Current Window Requests: {stats['current_window_requests']}")
        print(f"  Rate Limit: {stats['rate_limit']} requests/minute")
        print(f"  Requests Remaining: {stats['requests_remaining_in_window']}")
        print(f"  Window Time Remaining: {stats['rate_limit_window_remaining']:.1f} seconds")
        
        if stats['endpoint_usage']:
            print(f"\n🔗 Endpoint Usage:")
            for endpoint, count in sorted(stats['endpoint_usage'].items(), key=lambda x: x[1], reverse=True):
                print(f"  {endpoint}: {count} requests")
        
        # Test connection
        print(f"\n🧪 Testing API connection...")
        if client.test_connection():
            print("✅ API connection successful!")
        else:
            print("❌ API connection failed!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Set your API key in .env file")
        print("2. Installed required dependencies")
        print("3. Have internet connection")


if __name__ == "__main__":
    main()
