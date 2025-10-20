#!/usr/bin/env python3
"""
Database Setup Script for NCAA Football Analytics

This script sets up the database schema and loads data from CSV files.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.storage.database import create_database

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Main database setup function"""
    print("🗄️  NCAA Football Database Setup")
    print("=" * 50)
    
    try:
        # Create DuckDB database
        print("🔧 Creating DuckDB database...")
        db = create_database("duckdb")
        
        # Create schema
        print("📋 Creating database schema...")
        db.create_schema()
        
        # Load data from CSV
        csv_path = "data/models/ncaa_football_ml_dataset.csv"
        if Path(csv_path).exists():
            print(f"📊 Loading data from {csv_path}...")
            db.load_data_from_csv(csv_path)
            
            # Test database with sample queries
            print("\n🧪 Testing database with sample queries...")
            
            # Top 10 teams in 2024
            print("\n🏆 Top 10 Teams in 2024:")
            print("-" * 40)
            top_teams = db.get_top_teams(2024, 10)
            for _, row in top_teams.iterrows():
                print(f"   {row['team_name']} ({row['conference']}): {row['win_percentage']:.3f} win%")
            
            # Big Ten standings
            print("\n🏟️  Big Ten Conference Standings 2024:")
            print("-" * 50)
            standings = db.get_conference_standings(2024)
            big_ten = standings[standings['conference'] == 'B1G']
            for _, row in big_ten.head(10).iterrows():
                print(f"   {row['team_name']}: {row['wins']}-{row['losses']} ({row['win_percentage']:.3f})")
            
            # Team stats for Indiana and Ohio State
            print("\n🏈 Indiana and Ohio State Stats (2024):")
            print("-" * 45)
            for team in ['Indiana', 'Ohio State']:
                team_stats = db.get_team_stats(team, 2024)
                if not team_stats.empty:
                    stats = team_stats.iloc[0]
                    print(f"   {team}:")
                    print(f"     Record: {stats['wins']}-{stats['losses']} ({stats['win_percentage']:.3f})")
                    print(f"     Yards/Game: {stats.get('yards_per_game', 'N/A')}")
                    print(f"     Turnover Margin: {stats.get('turnover_margin', 'N/A')}")
            
            # Database statistics
            print("\n📈 Database Statistics:")
            print("-" * 25)
            
            # Count teams
            teams_count = db.query("SELECT COUNT(*) as count FROM teams").iloc[0]['count']
            print(f"   Teams: {teams_count}")
            
            # Count seasons
            seasons_count = db.query("SELECT COUNT(*) as count FROM seasons").iloc[0]['count']
            print(f"   Seasons: {seasons_count}")
            
            # Count team stats records
            stats_count = db.query("SELECT COUNT(*) as count FROM team_stats").iloc[0]['count']
            print(f"   Team Stats Records: {stats_count}")
            
            # Count derived metrics records
            metrics_count = db.query("SELECT COUNT(*) as count FROM derived_metrics").iloc[0]['count']
            print(f"   Derived Metrics Records: {metrics_count}")
            
            print("\n✅ Database setup completed successfully!")
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
