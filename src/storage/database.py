"""
Database Management for NCAA Football Analytics

Supports both DuckDB (embedded) and PostgreSQL (external) databases
with normalized schema for efficient data storage and querying.
"""

import os
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

try:
    import psycopg2
    from sqlalchemy import create_engine, text
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False
    print("⚠️  PostgreSQL not available, install with: pip install psycopg2-binary sqlalchemy")

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Main database management class"""
    
    def __init__(self, db_type: str = "duckdb", connection_string: Optional[str] = None):
        """
        Initialize database manager
        
        Args:
            db_type: "duckdb" or "postgresql"
            connection_string: Database connection string (for PostgreSQL)
        """
        self.db_type = db_type.lower()
        self.connection_string = connection_string
        self.engine = None
        self.connection = None
        self.db_path = None
        
        if self.db_type == "duckdb":
            self._setup_duckdb()
        elif self.db_type == "postgresql":
            self._setup_postgresql()
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    def _setup_duckdb(self):
        """Setup DuckDB connection"""
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB is not available. Install with: pip install duckdb")
        
        # Create database file in data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        self.db_path = data_dir / "ncaa_football.duckdb"
        self.connection = duckdb.connect(str(self.db_path))
        
        logger.info(f"DuckDB database initialized at: {self.db_path}")
    
    def _setup_postgresql(self):
        """Setup PostgreSQL connection"""
        if not POSTGRESQL_AVAILABLE:
            raise ImportError("PostgreSQL dependencies not available. Install with: pip install psycopg2-binary sqlalchemy")
        
        if not self.connection_string:
            # Use environment variables or default connection string
            self.connection_string = os.getenv(
                'DATABASE_URL',
                'postgresql://user:password@localhost:5432/ncaa_football'
            )
        
        self.engine = create_engine(self.connection_string)
        
        # Test connection
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("PostgreSQL connection established")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def create_schema(self):
        """Create normalized database schema"""
        logger.info("Creating database schema...")
        
        if self.db_type == "duckdb":
            self._create_duckdb_schema()
        elif self.db_type == "postgresql":
            self._create_postgresql_schema()
    
    def _create_duckdb_schema(self):
        """Create DuckDB schema"""
        schema_sql = """
        -- Teams table
        CREATE TABLE IF NOT EXISTS teams (
            team_id INTEGER PRIMARY KEY,
            team_name VARCHAR(100) NOT NULL,
            conference VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Seasons table
        CREATE TABLE IF NOT EXISTS seasons (
            season_id INTEGER PRIMARY KEY,
            year INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Team statistics table (main data)
        CREATE TABLE IF NOT EXISTS team_stats (
            id INTEGER PRIMARY KEY,
            team_id INTEGER,
            season_id INTEGER,
            games INTEGER,
            wins INTEGER,
            losses INTEGER,
            win_percentage DECIMAL(5,3),
            points_for INTEGER,
            points_against INTEGER,
            total_yards INTEGER,
            total_yards_opponent INTEGER,
            rushing_yards INTEGER,
            rushing_attempts INTEGER,
            rushing_tds INTEGER,
            net_passing_yards INTEGER,
            pass_completions INTEGER,
            pass_attempts INTEGER,
            passing_tds INTEGER,
            interceptions INTEGER,
            fumbles_recovered INTEGER,
            sacks INTEGER,
            tackles_for_loss INTEGER,
            third_down_conversions INTEGER,
            third_downs INTEGER,
            fourth_down_conversions INTEGER,
            fourth_downs INTEGER,
            penalties INTEGER,
            penalty_yards INTEGER,
            first_downs INTEGER,
            first_downs_opponent INTEGER,
            turnovers INTEGER,
            turnovers_opponent INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (team_id) REFERENCES teams(team_id),
            FOREIGN KEY (season_id) REFERENCES seasons(season_id)
        );
        
        -- Derived metrics table
        CREATE TABLE IF NOT EXISTS derived_metrics (
            id INTEGER PRIMARY KEY,
            team_id INTEGER,
            season_id INTEGER,
            yards_per_game DECIMAL(8,2),
            yards_allowed_per_game DECIMAL(8,2),
            turnover_margin INTEGER,
            first_down_differential INTEGER,
            offensive_efficiency DECIMAL(8,2),
            defensive_efficiency DECIMAL(8,2),
            rushing_efficiency DECIMAL(8,2),
            passing_efficiency DECIMAL(8,2),
            third_down_rate DECIMAL(5,3),
            defensive_pressure INTEGER,
            conference_strength DECIMAL(5,3),
            conference_rank INTEGER,
            conference_dominance DECIMAL(5,3),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (team_id) REFERENCES teams(team_id),
            FOREIGN KEY (season_id) REFERENCES seasons(season_id)
        );
        
        -- ML predictions table
        CREATE TABLE IF NOT EXISTS ml_predictions (
            id INTEGER PRIMARY KEY,
            team_id INTEGER,
            season_id INTEGER,
            prediction_year INTEGER,
            top_25_probability DECIMAL(5,3),
            predicted_top_25 BOOLEAN,
            predicted_rank DECIMAL(8,2),
            model_version VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (team_id) REFERENCES teams(team_id),
            FOREIGN KEY (season_id) REFERENCES seasons(season_id)
        );
        
        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_team_stats_team_season ON team_stats(team_id, season_id);
        CREATE INDEX IF NOT EXISTS idx_derived_metrics_team_season ON derived_metrics(team_id, season_id);
        CREATE INDEX IF NOT EXISTS idx_ml_predictions_team_season ON ml_predictions(team_id, season_id);
        CREATE INDEX IF NOT EXISTS idx_team_stats_season ON team_stats(season_id);
        CREATE INDEX IF NOT EXISTS idx_team_stats_conference ON team_stats(team_id);
        """
        
        self.connection.execute(schema_sql)
        logger.info("DuckDB schema created successfully")
    
    def _create_postgresql_schema(self):
        """Create PostgreSQL schema"""
        schema_sql = """
        -- Teams table
        CREATE TABLE IF NOT EXISTS teams (
            team_id SERIAL PRIMARY KEY,
            team_name VARCHAR(100) NOT NULL,
            conference VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Seasons table
        CREATE TABLE IF NOT EXISTS seasons (
            season_id SERIAL PRIMARY KEY,
            year INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Team statistics table (main data)
        CREATE TABLE IF NOT EXISTS team_stats (
            id SERIAL PRIMARY KEY,
            team_id INTEGER,
            season_id INTEGER,
            games INTEGER,
            wins INTEGER,
            losses INTEGER,
            win_percentage DECIMAL(5,3),
            points_for INTEGER,
            points_against INTEGER,
            total_yards INTEGER,
            total_yards_opponent INTEGER,
            rushing_yards INTEGER,
            rushing_attempts INTEGER,
            rushing_tds INTEGER,
            net_passing_yards INTEGER,
            pass_completions INTEGER,
            pass_attempts INTEGER,
            passing_tds INTEGER,
            interceptions INTEGER,
            fumbles_recovered INTEGER,
            sacks INTEGER,
            tackles_for_loss INTEGER,
            third_down_conversions INTEGER,
            third_downs INTEGER,
            fourth_down_conversions INTEGER,
            fourth_downs INTEGER,
            penalties INTEGER,
            penalty_yards INTEGER,
            first_downs INTEGER,
            first_downs_opponent INTEGER,
            turnovers INTEGER,
            turnovers_opponent INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (team_id) REFERENCES teams(team_id),
            FOREIGN KEY (season_id) REFERENCES seasons(season_id)
        );
        
        -- Derived metrics table
        CREATE TABLE IF NOT EXISTS derived_metrics (
            id SERIAL PRIMARY KEY,
            team_id INTEGER,
            season_id INTEGER,
            yards_per_game DECIMAL(8,2),
            yards_allowed_per_game DECIMAL(8,2),
            turnover_margin INTEGER,
            first_down_differential INTEGER,
            offensive_efficiency DECIMAL(8,2),
            defensive_efficiency DECIMAL(8,2),
            rushing_efficiency DECIMAL(8,2),
            passing_efficiency DECIMAL(8,2),
            third_down_rate DECIMAL(5,3),
            defensive_pressure INTEGER,
            conference_strength DECIMAL(5,3),
            conference_rank INTEGER,
            conference_dominance DECIMAL(5,3),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (team_id) REFERENCES teams(team_id),
            FOREIGN KEY (season_id) REFERENCES seasons(season_id)
        );
        
        -- ML predictions table
        CREATE TABLE IF NOT EXISTS ml_predictions (
            id SERIAL PRIMARY KEY,
            team_id INTEGER,
            season_id INTEGER,
            prediction_year INTEGER,
            top_25_probability DECIMAL(5,3),
            predicted_top_25 BOOLEAN,
            predicted_rank DECIMAL(8,2),
            model_version VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (team_id) REFERENCES teams(team_id),
            FOREIGN KEY (season_id) REFERENCES seasons(season_id)
        );
        
        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_team_stats_team_season ON team_stats(team_id, season_id);
        CREATE INDEX IF NOT EXISTS idx_derived_metrics_team_season ON derived_metrics(team_id, season_id);
        CREATE INDEX IF NOT EXISTS idx_ml_predictions_team_season ON ml_predictions(team_id, season_id);
        CREATE INDEX IF NOT EXISTS idx_team_stats_season ON team_stats(season_id);
        CREATE INDEX IF NOT EXISTS idx_team_stats_conference ON team_stats(team_id);
        """
        
        with self.engine.connect() as conn:
            conn.execute(text(schema_sql))
            conn.commit()
        
        logger.info("PostgreSQL schema created successfully")
    
    def load_data_from_csv(self, csv_path: str):
        """Load data from CSV files into the database"""
        logger.info(f"Loading data from CSV: {csv_path}")
        
        # Read the CSV
        df = pd.read_csv(csv_path)
        
        if self.db_type == "duckdb":
            self._load_data_duckdb(df)
        elif self.db_type == "postgresql":
            self._load_data_postgresql(df)
    
    def _load_data_duckdb(self, df: pd.DataFrame):
        """Load data into DuckDB"""
        # First, create lookup tables for teams and seasons
        teams_df = df[['team', 'conference']].drop_duplicates().reset_index(drop=True)
        teams_df['team_id'] = range(1, len(teams_df) + 1)
        
        seasons_df = df[['year']].drop_duplicates().reset_index(drop=True)
        seasons_df['season_id'] = range(1, len(seasons_df) + 1)
        
        # Insert teams and seasons
        self.connection.execute("DELETE FROM teams")
        for _, row in teams_df.iterrows():
            self.connection.execute("INSERT INTO teams (team_id, team_name, conference) VALUES (?, ?, ?)", 
                                  [row['team_id'], row['team'], row['conference']])
        
        self.connection.execute("DELETE FROM seasons")
        for _, row in seasons_df.iterrows():
            self.connection.execute("INSERT INTO seasons (season_id, year) VALUES (?, ?)", 
                                  [row['season_id'], row['year']])
        
        # Create team_id and season_id mappings
        team_mapping = dict(zip(teams_df['team'], teams_df['team_id']))
        season_mapping = dict(zip(seasons_df['year'], seasons_df['season_id']))
        
        # Prepare team_stats data
        df_clean = df.copy()
        df_clean['team_id'] = df_clean['team'].map(team_mapping)
        df_clean['season_id'] = df_clean['year'].map(season_mapping)
        
        # Map column names to database schema
        column_mapping = {
            'team': 'team_id',
            'year': 'season_id',
            'totalYards': 'total_yards',
            'totalYardsOpponent': 'total_yards_opponent',
            'rushingYards': 'rushing_yards',
            'rushingAttempts': 'rushing_attempts',
            'rushingTDs': 'rushing_tds',
            'netPassingYards': 'net_passing_yards',
            'passCompletions': 'pass_completions',
            'passAttempts': 'pass_attempts',
            'passingTDs': 'passing_tds',
            'sacks': 'sacks',
            'tacklesForLoss': 'tackles_for_loss',
            'thirdDownConversions': 'third_down_conversions',
            'thirdDowns': 'third_downs',
            'fourthDownConversions': 'fourth_down_conversions',
            'fourthDowns': 'fourth_downs',
            'firstDowns': 'first_downs',
            'firstDownsOpponent': 'first_downs_opponent',
            'turnovers': 'turnovers',
            'turnoversOpponent': 'turnovers_opponent'
        }
        
        # Select and rename columns
        stats_columns = ['team_id', 'season_id', 'games', 'wins', 'losses', 'win_percentage']
        for old_col, new_col in column_mapping.items():
            if old_col in df_clean.columns:
                stats_columns.append(new_col)
        
        team_stats_df = df_clean[stats_columns].copy()
        
        # Insert team stats
        self.connection.execute("DELETE FROM team_stats")
        for _, row in team_stats_df.iterrows():
            values = [row[col] for col in team_stats_df.columns]
            placeholders = ', '.join(['?' for _ in team_stats_df.columns])
            self.connection.execute(
                f"INSERT INTO team_stats ({', '.join(team_stats_df.columns)}) VALUES ({placeholders})",
                values
            )
        
        # Create derived metrics
        derived_df = df_clean[['team_id', 'season_id']].copy()
        
        # Calculate derived metrics
        if 'yards_per_game' in df_clean.columns:
            derived_df['yards_per_game'] = df_clean['yards_per_game']
        if 'yards_allowed_per_game' in df_clean.columns:
            derived_df['yards_allowed_per_game'] = df_clean['yards_allowed_per_game']
        if 'turnover_margin' in df_clean.columns:
            derived_df['turnover_margin'] = df_clean['turnover_margin']
        if 'first_down_differential' in df_clean.columns:
            derived_df['first_down_differential'] = df_clean['first_down_differential']
        if 'offensive_efficiency' in df_clean.columns:
            derived_df['offensive_efficiency'] = df_clean['offensive_efficiency']
        if 'defensive_efficiency' in df_clean.columns:
            derived_df['defensive_efficiency'] = df_clean['defensive_efficiency']
        if 'rushing_efficiency' in df_clean.columns:
            derived_df['rushing_efficiency'] = df_clean['rushing_efficiency']
        if 'passing_efficiency' in df_clean.columns:
            derived_df['passing_efficiency'] = df_clean['passing_efficiency']
        if 'third_down_rate' in df_clean.columns:
            derived_df['third_down_rate'] = df_clean['third_down_rate']
        if 'defensive_pressure' in df_clean.columns:
            derived_df['defensive_pressure'] = df_clean['defensive_pressure']
        if 'conference_strength' in df_clean.columns:
            derived_df['conference_strength'] = df_clean['conference_strength']
        if 'conference_rank' in df_clean.columns:
            derived_df['conference_rank'] = df_clean['conference_rank']
        if 'conference_dominance' in df_clean.columns:
            derived_df['conference_dominance'] = df_clean['conference_dominance']
        
        # Insert derived metrics
        self.connection.execute("DELETE FROM derived_metrics")
        if len(derived_df.columns) > 2:  # More than just team_id and season_id
            for _, row in derived_df.iterrows():
                values = [row[col] for col in derived_df.columns]
                placeholders = ', '.join(['?' for _ in derived_df.columns])
                self.connection.execute(
                    f"INSERT INTO derived_metrics ({', '.join(derived_df.columns)}) VALUES ({placeholders})",
                    values
                )
        
        logger.info("Data loaded into DuckDB successfully")
    
    def _load_data_postgresql(self, df: pd.DataFrame):
        """Load data into PostgreSQL"""
        # Similar to DuckDB but using SQLAlchemy
        # This is a simplified version - in production you'd want more robust error handling
        
        # Create lookup tables
        teams_df = df[['team', 'conference']].drop_duplicates().reset_index(drop=True)
        teams_df['team_id'] = range(1, len(teams_df) + 1)
        
        seasons_df = df[['year']].drop_duplicates().reset_index(drop=True)
        seasons_df['season_id'] = range(1, len(seasons_df) + 1)
        
        # Insert teams and seasons
        teams_df.to_sql('teams', self.engine, if_exists='replace', index=False)
        seasons_df.to_sql('seasons', self.engine, if_exists='replace', index=False)
        
        logger.info("Data loaded into PostgreSQL successfully")
    
    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame"""
        if self.db_type == "duckdb":
            return self.connection.execute(sql).df()
        elif self.db_type == "postgresql":
            return pd.read_sql(sql, self.engine)
    
    def get_team_stats(self, team_name: Optional[str] = None, season: Optional[int] = None) -> pd.DataFrame:
        """Get team statistics with optional filters"""
        sql = """
        SELECT 
            t.team_name,
            t.conference,
            s.year,
            ts.*,
            dm.*
        FROM team_stats ts
        JOIN teams t ON ts.team_id = t.team_id
        JOIN seasons s ON ts.season_id = s.season_id
        LEFT JOIN derived_metrics dm ON ts.team_id = dm.team_id AND ts.season_id = dm.season_id
        """
        
        conditions = []
        if team_name:
            conditions.append(f"t.team_name = '{team_name}'")
        if season:
            conditions.append(f"s.year = {season}")
        
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        
        sql += " ORDER BY s.year DESC, ts.win_percentage DESC"
        
        return self.query(sql)
    
    def get_conference_standings(self, season: int) -> pd.DataFrame:
        """Get conference standings for a specific season"""
        sql = f"""
        SELECT 
            t.team_name,
            t.conference,
            ts.wins,
            ts.losses,
            ts.win_percentage,
            ts.total_yards,
            ts.total_yards_opponent,
            dm.yards_per_game,
            dm.yards_allowed_per_game,
            dm.turnover_margin
        FROM team_stats ts
        JOIN teams t ON ts.team_id = t.team_id
        JOIN seasons s ON ts.season_id = s.season_id
        LEFT JOIN derived_metrics dm ON ts.team_id = dm.team_id AND ts.season_id = dm.season_id
        WHERE s.year = {season}
        ORDER BY t.conference, ts.win_percentage DESC
        """
        
        return self.query(sql)
    
    def get_top_teams(self, season: int, limit: int = 25) -> pd.DataFrame:
        """Get top teams by win percentage for a season"""
        sql = f"""
        SELECT 
            t.team_name,
            t.conference,
            ts.wins,
            ts.losses,
            ts.win_percentage,
            dm.yards_per_game,
            dm.turnover_margin
        FROM team_stats ts
        JOIN teams t ON ts.team_id = t.team_id
        JOIN seasons s ON ts.season_id = s.season_id
        LEFT JOIN derived_metrics dm ON ts.team_id = dm.team_id AND ts.season_id = dm.season_id
        WHERE s.year = {season}
        ORDER BY ts.win_percentage DESC
        LIMIT {limit}
        """
        
        return self.query(sql)
    
    def close(self):
        """Close database connection"""
        if self.db_type == "duckdb" and self.connection:
            self.connection.close()
        elif self.db_type == "postgresql" and self.engine:
            self.engine.dispose()
        
        logger.info("Database connection closed")

def create_database(db_type: str = "duckdb", connection_string: Optional[str] = None) -> DatabaseManager:
    """Factory function to create database manager"""
    return DatabaseManager(db_type, connection_string)

if __name__ == "__main__":
    # Example usage
    db = create_database("duckdb")
    db.create_schema()
    
    # Load data if available
    csv_path = "data/models/ncaa_football_ml_dataset.csv"
    if Path(csv_path).exists():
        db.load_data_from_csv(csv_path)
        
        # Test queries
        print("Top 10 teams in 2024:")
        print(db.get_top_teams(2024, 10))
        
        print("\nBig Ten standings 2024:")
        standings = db.get_conference_standings(2024)
        big_ten = standings[standings['conference'] == 'B1G']
        print(big_ten.head(10))
    
    db.close()
