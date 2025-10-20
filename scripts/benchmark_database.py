#!/usr/bin/env python3
"""
Database Performance Comparison Script

This script compares the performance of database queries vs CSV file operations
to demonstrate the benefits of database integration.
"""

import sys
import time
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.storage.simple_database import create_simple_database

def benchmark_csv_operations():
    """Benchmark CSV file operations"""
    print("📊 Benchmarking CSV Operations...")
    
    csv_path = "data/models/ncaa_football_ml_dataset.csv"
    if not Path(csv_path).exists():
        print("❌ CSV file not found")
        return None
    
    start_time = time.time()
    
    # Load CSV
    df = pd.read_csv(csv_path)
    load_time = time.time() - start_time
    
    # Filter operations
    filter_start = time.time()
    
    # Filter by season
    df_2024 = df[df['year'] == 2024]
    
    # Filter by conference
    big_ten = df_2024[df_2024['conference'] == 'B1G']
    
    # Sort by win percentage
    sorted_teams = big_ten.sort_values('win_percentage', ascending=False)
    
    # Get top teams
    top_teams = df_2024.sort_values('win_percentage', ascending=False).head(10)
    
    filter_time = time.time() - filter_start
    
    total_time = time.time() - start_time
    
    return {
        'load_time': load_time,
        'filter_time': filter_time,
        'total_time': total_time,
        'records_processed': len(df),
        'big_ten_records': len(big_ten),
        'top_teams': len(top_teams)
    }

def benchmark_database_operations():
    """Benchmark database operations"""
    print("🗄️  Benchmarking Database Operations...")
    
    try:
        db = create_simple_database()
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return None
    
    start_time = time.time()
    
    # Database queries
    query_start = time.time()
    
    # Get 2024 data
    df_2024 = db.query("SELECT * FROM ncaa_football_data WHERE year = 2024")
    
    # Get Big Ten teams
    big_ten = db.get_big_ten_teams(2024)
    
    # Get top teams
    top_teams = db.get_top_teams(2024, 10)
    
    query_time = time.time() - query_start
    
    total_time = time.time() - start_time
    
    db.close()
    
    return {
        'query_time': query_time,
        'total_time': total_time,
        'records_processed': len(df_2024),
        'big_ten_records': len(big_ten),
        'top_teams': len(top_teams)
    }

def main():
    """Main benchmark function"""
    print("🏈 NCAA Football Analytics - Database Performance Comparison")
    print("=" * 70)
    
    # Benchmark CSV operations
    csv_results = benchmark_csv_operations()
    
    if csv_results:
        print(f"\n📊 CSV Performance Results:")
        print(f"   Load Time: {csv_results['load_time']:.4f} seconds")
        print(f"   Filter Time: {csv_results['filter_time']:.4f} seconds")
        print(f"   Total Time: {csv_results['total_time']:.4f} seconds")
        print(f"   Records Processed: {csv_results['records_processed']:,}")
        print(f"   Big Ten Records: {csv_results['big_ten_records']}")
        print(f"   Top Teams Retrieved: {csv_results['top_teams']}")
    
    print("\n" + "="*70)
    
    # Benchmark database operations
    db_results = benchmark_database_operations()
    
    if db_results:
        print(f"\n🗄️  Database Performance Results:")
        print(f"   Query Time: {db_results['query_time']:.4f} seconds")
        print(f"   Total Time: {db_results['total_time']:.4f} seconds")
        print(f"   Records Processed: {db_results['records_processed']:,}")
        print(f"   Big Ten Records: {db_results['big_ten_records']}")
        print(f"   Top Teams Retrieved: {db_results['top_teams']}")
    
    # Performance comparison
    if csv_results and db_results:
        print(f"\n🚀 Performance Comparison:")
        print("-" * 30)
        
        speedup = csv_results['total_time'] / db_results['total_time']
        print(f"   Database is {speedup:.2f}x faster than CSV")
        
        if csv_results['filter_time'] > 0:
            filter_speedup = csv_results['filter_time'] / db_results['query_time']
            print(f"   Database queries are {filter_speedup:.2f}x faster than CSV filtering")
        
        print(f"\n💡 Benefits of Database Integration:")
        print(f"   • Faster data access and filtering")
        print(f"   • Reduced memory usage (no need to load entire CSV)")
        print(f"   • Better scalability for larger datasets")
        print(f"   • SQL query capabilities for complex analytics")
        print(f"   • Indexed queries for optimal performance")
    
    print(f"\n✅ Performance comparison complete!")

if __name__ == "__main__":
    main()
