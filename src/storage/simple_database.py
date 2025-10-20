"""
Simplified Database Management for NCAA Football Analytics

A streamlined approach using DuckDB with the actual data structure.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime

# Database imports
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    print("⚠️  DuckDB not available, install with: pip install duckdb")

logger = logging.getLogger(__name__)

class SimpleDatabaseManager:
    """Simplified database management class"""
    
    def __init__(self):
        """Initialize DuckDB connection"""
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB is not available. Install with: pip install duckdb")
        
        # Create database file in data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        self.db_path = data_dir / "ncaa_football_simple.duckdb"
        self.connection = duckdb.connect(str(self.db_path))
        
        logger.info(f"Simple DuckDB database initialized at: {self.db_path}")
    
    def create_schema_from_csv(self, csv_path: str):
        """Create schema automatically from CSV file"""
        logger.info("Creating schema from CSV file...")
        
        # Read CSV to get column info
        df = pd.read_csv(csv_path)
        
        # Drop existing tables if they exist
        self.connection.execute("DROP TABLE IF EXISTS ncaa_football_data")
        
        # Create table with all columns from CSV
        # Use DuckDB's ability to infer schema from CSV
        self.connection.execute(f"CREATE TABLE ncaa_football_data AS SELECT * FROM '{csv_path}' WHERE 1=0")
        
        # Create indexes for better performance
        index_sql = """
        CREATE INDEX IF NOT EXISTS idx_team_year ON ncaa_football_data(team, year);
        CREATE INDEX IF NOT EXISTS idx_conference_year ON ncaa_football_data(conference, year);
        CREATE INDEX IF NOT EXISTS idx_year ON ncaa_football_data(year);
        """
        
        # Only create indexes for columns that exist
        try:
            self.connection.execute(index_sql)
        except Exception as e:
            logger.warning(f"Could not create some indexes: {e}")
        
        logger.info(f"Schema created with {len(df.columns)} columns")
    
    def create_schema(self):
        """Create simplified database schema (legacy method)"""
        logger.info("Creating simplified database schema...")
        
        # Drop existing tables if they exist
        self.connection.execute("DROP TABLE IF EXISTS ncaa_football_data")
        
        # Create a single table with all the data
        schema_sql = """
        CREATE TABLE ncaa_football_data (
            team VARCHAR(100),
            year INTEGER,
            conference VARCHAR(50),
            is_national_champion INTEGER,
            conference_champion VARCHAR(50),
            games INTEGER,
            win_percentage DECIMAL(5,3),
            
            -- Basic stats
            totalYards INTEGER,
            totalYardsOpponent INTEGER,
            rushingYards INTEGER,
            rushingAttempts INTEGER,
            rushingTDs INTEGER,
            netPassingYards INTEGER,
            passCompletions INTEGER,
            passAttempts INTEGER,
            passingTDs INTEGER,
            
            -- Defensive stats
            sacks INTEGER,
            tacklesForLoss INTEGER,
            interceptions INTEGER,
            fumblesRecovered INTEGER,
            
            -- Special teams and efficiency
            thirdDownConversions INTEGER,
            thirdDowns INTEGER,
            fourthDownConversions INTEGER,
            fourthDowns INTEGER,
            penalties INTEGER,
            penaltyYards INTEGER,
            
            -- Team stats
            firstDowns INTEGER,
            firstDownsOpponent INTEGER,
            turnovers INTEGER,
            turnoversOpponent INTEGER,
            
            -- Derived metrics
            yards_per_game DECIMAL(8,2),
            yards_allowed_per_game DECIMAL(8,2),
            yard_differential INTEGER,
            turnover_margin INTEGER,
            first_down_differential INTEGER,
            third_down_conversion_rate DECIMAL(5,3),
            fourth_down_conversion_rate DECIMAL(5,3),
            rushing_yards_per_attempt DECIMAL(8,2),
            passing_yards_per_attempt DECIMAL(8,2),
            completion_percentage DECIMAL(5,3),
            offensive_efficiency DECIMAL(8,2),
            defensive_efficiency DECIMAL(8,2),
            
            -- Advanced metrics
            pythagorean_expectation DECIMAL(5,3),
            margin_of_victory DECIMAL(8,2),
            turnovers_per_game DECIMAL(5,2),
            turnover_margin_per_game DECIMAL(5,2),
            offensive_balance DECIMAL(5,3),
            defensive_pressure INTEGER,
            
            -- Conference metrics
            conference_strength DECIMAL(5,3),
            conference_rank INTEGER,
            conference_dominance DECIMAL(5,3),
            
            -- Historical metrics
            prev_year_win_pct DECIMAL(5,3),
            prev_year_yards_per_game DECIMAL(8,2),
            prev_year_yard_differential INTEGER,
            win_pct_3yr DECIMAL(5,3),
            yards_per_game_3yr DECIMAL(8,2),
            win_pct_trend DECIMAL(5,3),
            
            -- Metadata
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        self.connection.execute(schema_sql)
        
        # Create indexes for better performance
        index_sql = """
        CREATE INDEX IF NOT EXISTS idx_team_year ON ncaa_football_data(team, year);
        CREATE INDEX IF NOT EXISTS idx_conference_year ON ncaa_football_data(conference, year);
        CREATE INDEX IF NOT EXISTS idx_year ON ncaa_football_data(year);
        CREATE INDEX IF NOT EXISTS idx_win_percentage ON ncaa_football_data(win_percentage);
        """
        
        self.connection.execute(index_sql)
        logger.info("Simplified database schema created successfully")
    
    def load_data_from_csv(self, csv_path: str):
        """Load data from CSV file"""
        logger.info(f"Loading data from CSV: {csv_path}")
        
        # Use DuckDB's built-in CSV import capability
        self.connection.execute("DELETE FROM ncaa_football_data")
        
        # Import CSV directly into table
        import_sql = f"INSERT INTO ncaa_football_data SELECT * FROM '{csv_path}'"
        self.connection.execute(import_sql)
        
        # Get count of loaded records
        result = self.connection.execute("SELECT COUNT(*) as count FROM ncaa_football_data").fetchone()
        count = result[0] if result else 0
        
        logger.info(f"Data loaded successfully: {count} records")
    
    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame"""
        return self.connection.execute(sql).df()
    
    def get_team_stats(self, team_name: Optional[str] = None, season: Optional[int] = None) -> pd.DataFrame:
        """Get team statistics with optional filters"""
        sql = "SELECT * FROM ncaa_football_data"
        
        conditions = []
        if team_name:
            conditions.append(f"team = '{team_name}'")
        if season:
            conditions.append(f"year = {season}")
        
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        
        sql += " ORDER BY year DESC, win_percentage DESC"
        
        return self.query(sql)
    
    def get_conference_standings(self, season: int) -> pd.DataFrame:
        """Get conference standings for a specific season"""
        sql = f"""
        SELECT 
            team,
            conference,
            games,
            win_percentage,
            totalYards,
            totalYardsOpponent,
            yards_per_game,
            yards_allowed_per_game,
            turnover_margin
        FROM ncaa_football_data
        WHERE year = {season}
        ORDER BY conference, win_percentage DESC
        """
        
        return self.query(sql)
    
    def get_top_teams(self, season: int, limit: int = 25) -> pd.DataFrame:
        """Get top teams by win percentage for a season"""
        sql = f"""
        SELECT 
            team,
            conference,
            win_percentage,
            yards_per_game,
            turnover_margin,
            offensive_efficiency,
            defensive_efficiency
        FROM ncaa_football_data
        WHERE year = {season}
        ORDER BY win_percentage DESC
        LIMIT {limit}
        """
        
        return self.query(sql)
    
    def get_big_ten_teams(self, season: int) -> pd.DataFrame:
        """Get Big Ten teams for a specific season"""
        sql = f"""
        SELECT 
            team,
            conference,
            win_percentage,
            yards_per_game,
            yards_allowed_per_game,
            turnover_margin,
            offensive_efficiency,
            defensive_efficiency
        FROM ncaa_football_data
        WHERE year = {season} AND conference = 'B1G'
        ORDER BY win_percentage DESC
        """
        
        return self.query(sql)
    
    def get_team_comparison(self, team1: str, team2: str, season: int) -> pd.DataFrame:
        """Compare two teams for a specific season"""
        sql = f"""
        SELECT 
            team,
            conference,
            win_percentage,
            yards_per_game,
            yards_allowed_per_game,
            turnover_margin,
            offensive_efficiency,
            defensive_efficiency,
            pythagorean_expectation
        FROM ncaa_football_data
        WHERE year = {season} AND team IN ('{team1}', '{team2}')
        ORDER BY win_percentage DESC
        """
        
        return self.query(sql)
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        stats = {}
        
        # Total records
        result = self.query("SELECT COUNT(*) as count FROM ncaa_football_data")
        stats['total_records'] = result.iloc[0]['count']
        
        # Unique teams
        result = self.query("SELECT COUNT(DISTINCT team) as count FROM ncaa_football_data")
        stats['unique_teams'] = result.iloc[0]['count']
        
        # Seasons
        result = self.query("SELECT COUNT(DISTINCT year) as count FROM ncaa_football_data")
        stats['seasons'] = result.iloc[0]['count']
        
        # Conferences
        result = self.query("SELECT COUNT(DISTINCT conference) as count FROM ncaa_football_data")
        stats['conferences'] = result.iloc[0]['count']
        
        return stats
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
        logger.info("Database connection closed")

def create_simple_database() -> SimpleDatabaseManager:
    """Factory function to create simple database manager"""
    return SimpleDatabaseManager()

if __name__ == "__main__":
    # Example usage
    db = create_simple_database()
    db.create_schema()
    
    # Load data if available
    csv_path = "data/models/ncaa_football_ml_dataset.csv"
    if Path(csv_path).exists():
        db.load_data_from_csv(csv_path)
        
        # Test queries
        print("Top 10 teams in 2024:")
        print(db.get_top_teams(2024, 10))
        
        print("\nBig Ten standings 2024:")
        big_ten = db.get_big_ten_teams(2024)
        print(big_ten)
        
        print("\nIndiana vs Ohio State comparison:")
        comparison = db.get_team_comparison('Indiana', 'Ohio State', 2024)
        print(comparison)
    
    db.close()
