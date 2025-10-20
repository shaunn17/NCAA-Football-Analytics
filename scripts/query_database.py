#!/usr/bin/env python3
"""
Database Query Tool for NCAA Football Analytics

A simple command-line tool to query the database and explore data.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.storage.simple_database import create_simple_database

def main():
    """Main query tool function"""
    parser = argparse.ArgumentParser(
        description="Query NCAA Football Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get top teams for 2024
  python scripts/query_database.py --top-teams 2024 --limit 10
  
  # Get Big Ten standings
  python scripts/query_database.py --big-ten 2024
  
  # Compare two teams
  python scripts/query_database.py --compare Indiana Ohio-State --season 2024
  
  # Custom SQL query
  python scripts/query_database.py --sql "SELECT team, win_percentage FROM ncaa_football_data WHERE year = 2024 ORDER BY win_percentage DESC LIMIT 5"
        """
    )
    
    parser.add_argument('--top-teams', type=int, help='Get top teams for a specific year')
    parser.add_argument('--big-ten', type=int, help='Get Big Ten standings for a specific year')
    parser.add_argument('--compare', nargs=2, metavar=('TEAM1', 'TEAM2'), help='Compare two teams')
    parser.add_argument('--season', type=int, help='Season for comparison (default: 2024)')
    parser.add_argument('--limit', type=int, default=10, help='Limit for top teams query (default: 10)')
    parser.add_argument('--sql', type=str, help='Custom SQL query')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    
    args = parser.parse_args()
    
    try:
        # Connect to database
        print("🗄️  Connecting to NCAA Football Database...")
        db = create_simple_database()
        
        # Database statistics
        if args.stats:
            print("\n📊 Database Statistics:")
            print("-" * 30)
            stats = db.get_database_stats()
            for key, value in stats.items():
                print(f"   {key.replace('_', ' ').title()}: {value}")
        
        # Top teams query
        if args.top_teams:
            print(f"\n🏆 Top {args.limit} Teams in {args.top_teams}:")
            print("-" * 40)
            top_teams = db.get_top_teams(args.top_teams, args.limit)
            for _, row in top_teams.iterrows():
                team = row.get('team', 'Unknown')
                conf = row.get('conference', 'Unknown')
                win_pct = row.get('win_percentage', 0) or 0
                yards_pg = row.get('yards_per_game', 0) or 0
                print(f"   {team} ({conf}): {win_pct:.3f} win%, {yards_pg:.1f} yards/game")
        
        # Big Ten standings
        if args.big_ten:
            print(f"\n🏟️  Big Ten Conference Standings {args.big_ten}:")
            print("-" * 50)
            big_ten = db.get_big_ten_teams(args.big_ten)
            for _, row in big_ten.iterrows():
                team = row.get('team', 'Unknown')
                win_pct = row.get('win_percentage', 0) or 0
                yards_pg = row.get('yards_per_game', 0) or 0
                yards_allowed = row.get('yards_allowed_per_game', 0) or 0
                print(f"   {team}: {win_pct:.3f} win%, {yards_pg:.1f} off, {yards_allowed:.1f} def")
        
        # Team comparison
        if args.compare:
            season = args.season or 2024
            team1, team2 = args.compare
            print(f"\n🏈 {team1} vs {team2} Comparison ({season}):")
            print("-" * 45)
            comparison = db.get_team_comparison(team1, team2, season)
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
        
        # Custom SQL query
        if args.sql:
            print(f"\n🔍 Custom Query Results:")
            print("-" * 30)
            result = db.query(args.sql)
            print(result.to_string(index=False))
        
        # If no specific query, show some basic info
        if not any([args.top_teams, args.big_ten, args.compare, args.sql, args.stats]):
            print("\n📋 Available Data:")
            print("-" * 20)
            stats = db.get_database_stats()
            print(f"   Total Records: {stats['total_records']}")
            print(f"   Unique Teams: {stats['unique_teams']}")
            print(f"   Seasons: {stats['seasons']}")
            print(f"   Conferences: {stats['conferences']}")
            
            print("\n💡 Use --help to see available query options")
        
        # Close database connection
        db.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
