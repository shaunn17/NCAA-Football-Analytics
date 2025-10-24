#!/usr/bin/env python3
"""
API Usage Monitor for College Football Data API

This script helps you track and monitor your API usage across different methods:
1. Built-in request counting
2. Log file analysis
3. API response headers
4. Usage statistics and reporting
"""

import sys
import os
from pathlib import Path
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.api_client import CollegeFootballDataAPIClient
from config.settings import settings


class APIUsageMonitor:
    """Monitor and track API usage"""
    
    def __init__(self):
        self.api_client = CollegeFootballDataAPIClient()
        self.usage_log_file = Path("api_usage_log.json")
        self.load_usage_log()
    
    def load_usage_log(self):
        """Load existing usage log"""
        if self.usage_log_file.exists():
            with open(self.usage_log_file, 'r') as f:
                self.usage_log = json.load(f)
        else:
            self.usage_log = {
                "total_requests": 0,
                "daily_usage": {},
                "endpoint_usage": {},
                "rate_limit_hits": 0,
                "last_reset": datetime.now().isoformat()
            }
    
    def save_usage_log(self):
        """Save usage log to file"""
        with open(self.usage_log_file, 'w') as f:
            json.dump(self.usage_log, f, indent=2)
    
    def log_request(self, endpoint: str, params: Dict = None):
        """Log an API request"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Update total requests
        self.usage_log["total_requests"] += 1
        
        # Update daily usage
        if today not in self.usage_log["daily_usage"]:
            self.usage_log["daily_usage"][today] = 0
        self.usage_log["daily_usage"][today] += 1
        
        # Update endpoint usage
        if endpoint not in self.usage_log["endpoint_usage"]:
            self.usage_log["endpoint_usage"][endpoint] = 0
        self.usage_log["endpoint_usage"][endpoint] += 1
        
        self.save_usage_log()
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current API usage statistics"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        return {
            "total_requests": self.usage_log["total_requests"],
            "today_requests": self.usage_log["daily_usage"].get(today, 0),
            "rate_limit": settings.api_rate_limit,
            "rate_limit_window": "1 minute",
            "current_client_requests": self.api_client.request_count,
            "last_reset": self.usage_log["last_reset"]
        }
    
    def get_daily_usage_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get daily usage summary for the last N days"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        daily_summary = {}
        total_requests = 0
        
        for i in range(days):
            date = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
            requests = self.usage_log["daily_usage"].get(date, 0)
            daily_summary[date] = requests
            total_requests += requests
        
        return {
            "period_days": days,
            "total_requests": total_requests,
            "average_per_day": total_requests / days,
            "daily_breakdown": daily_summary
        }
    
    def get_endpoint_usage(self) -> Dict[str, Any]:
        """Get usage breakdown by endpoint"""
        total = self.usage_log["total_requests"]
        endpoint_usage = {}
        
        for endpoint, count in self.usage_log["endpoint_usage"].items():
            endpoint_usage[endpoint] = {
                "requests": count,
                "percentage": (count / total * 100) if total > 0 else 0
            }
        
        return endpoint_usage
    
    def analyze_log_files(self) -> Dict[str, Any]:
        """Analyze log files for API usage patterns"""
        log_files = list(Path(".").glob("*.log"))
        api_requests = 0
        rate_limit_hits = 0
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    
                # Count API requests
                api_requests += len(re.findall(r"Making request to", content))
                
                # Count rate limit hits
                rate_limit_hits += len(re.findall(r"Rate limit reached", content))
                
            except Exception as e:
                print(f"Error reading {log_file}: {e}")
        
        return {
            "log_files_analyzed": len(log_files),
            "api_requests_found": api_requests,
            "rate_limit_hits_found": rate_limit_hits
        }
    
    def print_usage_report(self):
        """Print comprehensive usage report"""
        print("🏈 College Football Data API Usage Report")
        print("=" * 50)
        
        # Current usage
        current = self.get_current_usage()
        print(f"\n📊 Current Usage:")
        print(f"  Total Requests: {current['total_requests']}")
        print(f"  Today's Requests: {current['today_requests']}")
        print(f"  Rate Limit: {current['rate_limit']} requests/minute")
        print(f"  Current Client Requests: {current['current_client_requests']}")
        
        # Daily summary
        daily = self.get_daily_usage_summary(7)
        print(f"\n📅 Last 7 Days Summary:")
        print(f"  Total Requests: {daily['total_requests']}")
        print(f"  Average per Day: {daily['average_per_day']:.1f}")
        print(f"  Daily Breakdown:")
        for date, requests in daily['daily_breakdown'].items():
            print(f"    {date}: {requests} requests")
        
        # Endpoint usage
        endpoints = self.get_endpoint_usage()
        if endpoints:
            print(f"\n🔗 Endpoint Usage:")
            for endpoint, usage in sorted(endpoints.items(), key=lambda x: x[1]['requests'], reverse=True):
                print(f"  {endpoint}: {usage['requests']} requests ({usage['percentage']:.1f}%)")
        
        # Log file analysis
        log_analysis = self.analyze_log_files()
        print(f"\n📝 Log File Analysis:")
        print(f"  Files Analyzed: {log_analysis['log_files_analyzed']}")
        print(f"  API Requests Found: {log_analysis['api_requests_found']}")
        print(f"  Rate Limit Hits: {log_analysis['rate_limit_hits_found']}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if current['today_requests'] > current['rate_limit'] * 10:  # More than 10 minutes worth
            print("  ⚠️  High usage detected - consider optimizing requests")
        if daily['average_per_day'] > current['rate_limit'] * 60:  # More than 1 hour worth per day
            print("  ⚠️  Consider implementing request caching")
        print("  ✅ Monitor rate limits to avoid hitting API limits")
        print("  ✅ Use the built-in rate limiting in the API client")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Monitor College Football Data API Usage")
    parser.add_argument("--report", action="store_true", help="Show usage report")
    parser.add_argument("--daily", type=int, default=7, help="Days for daily summary (default: 7)")
    parser.add_argument("--test", action="store_true", help="Test API connection and log a request")
    
    args = parser.parse_args()
    
    monitor = APIUsageMonitor()
    
    if args.test:
        print("🧪 Testing API connection...")
        try:
            # Test connection
            teams = monitor.api_client.get_teams()
            monitor.log_request("teams")
            print(f"✅ API connection successful! Found {len(teams)} teams")
            print(f"📊 Request logged to usage tracker")
        except Exception as e:
            print(f"❌ API test failed: {e}")
    
    if args.report or not any([args.test]):
        monitor.print_usage_report()


if __name__ == "__main__":
    main()
