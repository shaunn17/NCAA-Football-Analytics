#!/usr/bin/env python3
"""
Simplified Database Setup Script for NCAA Football Analytics

This script sets up a simplified database schema and loads data from CSV files.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.storage.simple_database import create_simple_database

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Main database setup function"""
    print("🗄️  NCAA Football Simplified Database Setup")
    print("=" * 50)
    
    try:
        # Create simplified database
        print("🔧 Creating simplified DuckDB database...")
        db = create_simple_database()
        
        # Create schema from CSV
        csv_path = "data/models/ncaa_football_ml_dataset.csv"
        if Path(csv_path).exists():
            print(f"📋 Creating database schema from CSV...")
            db.create_schema_from_csv(csv_path)
            
            print(f"📊 Loading data from {csv_path}...")
            db.load_data_from_csv(csv_path)
            
            # Test database with sample queries
            print("\n🧪 Testing database with sample queries...")
            
            # Database statistics
            stats = db.get_database_stats()
            print(f"\n📈 Database Statistics:")
            print(f"   Total Records: {stats['total_records']}")
            print(f"   Unique Teams: {stats['unique_teams']}")
            print(f"   Seasons: {stats['seasons']}")
            print(f"   Conferences: {stats['conferences']}")
            
            # Top 10 teams in 2024
            print("\n🏆 Top 10 Teams in 2024:")
            print("-" * 40)
            top_teams = db.get_top_teams(2024, 10)
            for _, row in top_teams.iterrows():
                win_pct = row.get('win_percentage', 0) or 0
                team = row.get('team', 'Unknown')
                conf = row.get('conference', 'Unknown')
                print(f"   {team} ({conf}): {win_pct:.3f} win%")
            
            # Big Ten standings
            print("\n🏟️  Big Ten Conference Standings 2024:")
            print("-" * 50)
            big_ten = db.get_big_ten_teams(2024)
            for _, row in big_ten.iterrows():
                win_pct = row.get('win_percentage', 0) or 0
                yards_pg = row.get('yards_per_game', 0) or 0
                team = row.get('team', 'Unknown')
                print(f"   {team}: {win_pct:.3f} win%, {yards_pg:.1f} yards/game")
            
            # Team comparison: Indiana vs Ohio State
            print("\n🏈 Indiana vs Ohio State Comparison (2024):")
            print("-" * 45)
            comparison = db.get_team_comparison('Indiana', 'Ohio State', 2024)
            for _, row in comparison.iterrows():
                team = row.get('team', 'Unknown')
                win_pct = row.get('win_percentage', 0) or 0
                yards_pg = row.get('yards_per_game', 0) or 0
                yards_allowed = row.get('yards_allowed_per_game', 0) or 0
                turnover_margin = row.get('turnover_margin', 0) or 0
                off_eff = row.get('offensive_efficiency', 0) or 0
                def_eff = row.get('defensive_efficiency', 0) or 0
                
                print(f"   {team}:")
                print(f"     Win%: {win_pct:.3f}")
                print(f"     Yards/Game: {yards_pg:.1f}")
                print(f"     Yards Allowed/Game: {yards_allowed:.1f}")
                print(f"     Turnover Margin: {turnover_margin}")
                print(f"     Offensive Efficiency: {off_eff:.1f}")
                print(f"     Defensive Efficiency: {def_eff:.1f}")
                print()
            
            print("✅ Database setup completed successfully!")
            print(f"📁 Database file: {db.db_path}")
            
        else:
            print(f"⚠️  CSV file not found: {csv_path}")
            print("Please run the data pipeline first to generate the CSV file.")
            return 1
        
        # Close database connection
        db.close()
        
    except Exception as e:
        print(f"❌ Error during database setup: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
