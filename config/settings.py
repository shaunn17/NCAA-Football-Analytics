"""
Configuration settings for the NCAA Football Analytics Platform
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Configuration
    college_football_data_api_key: str = Field(
        default="",
        description="College Football Data API key"
    )
    api_base_url: str = Field(
        default="https://api.collegefootballdata.com",
        description="Base URL for College Football Data API"
    )
    api_timeout: int = Field(
        default=30,
        description="API request timeout in seconds"
    )
    api_rate_limit: int = Field(
        default=100,
        description="API requests per minute limit"
    )
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql://localhost:5432/ncaa_football",
        description="Database connection URL"
    )
    database_echo: bool = Field(
        default=False,
        description="Enable SQLAlchemy query logging"
    )
    
    # DuckDB Configuration (alternative to PostgreSQL)
    use_duckdb: bool = Field(
        default=False,
        description="Use DuckDB instead of PostgreSQL"
    )
    duckdb_path: str = Field(
        default="data/ncaa_football.db",
        description="Path to DuckDB database file"
    )
    
    # Data Paths
    project_root: Path = Field(
        default=Path(__file__).parent.parent,
        description="Project root directory"
    )
    data_dir: Path = Field(
        default=Path(__file__).parent.parent / "data",
        description="Data directory"
    )
    raw_data_dir: Path = Field(
        default=Path(__file__).parent.parent / "data" / "raw",
        description="Raw data directory"
    )
    processed_data_dir: Path = Field(
        default=Path(__file__).parent.parent / "data" / "processed",
        description="Processed data directory"
    )
    models_dir: Path = Field(
        default=Path(__file__).parent.parent / "data" / "models",
        description="Models directory"
    )
    
    # Data Collection Settings
    seasons_to_collect: list[int] = Field(
        default=[2018, 2019, 2020, 2021, 2022, 2023, 2024],
        description="List of seasons to collect data for"
    )
    conferences_to_focus: list[str] = Field(
        default=[
            "Big Ten", "SEC", "ACC", "Big 12", "Pac-12",
            "American Athletic", "Mountain West", "MAC", "Sun Belt", "Conference USA"
        ],
        description="List of conferences to focus on"
    )
    
    # Machine Learning Settings
    ml_random_state: int = Field(
        default=42,
        description="Random state for ML models"
    )
    train_test_split: float = Field(
        default=0.8,
        description="Train/test split ratio"
    )
    cv_folds: int = Field(
        default=5,
        description="Number of cross-validation folds"
    )
    
    # Dashboard Settings
    dashboard_host: str = Field(
        default="127.0.0.1",
        description="Dashboard host"
    )
    dashboard_port: int = Field(
        default=8050,
        description="Dashboard port"
    )
    dashboard_debug: bool = Field(
        default=True,
        description="Enable dashboard debug mode"
    )
    
    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Log file path (None for console only)"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Create settings instance
settings = Settings()

# Ensure directories exist
settings.data_dir.mkdir(exist_ok=True)
settings.raw_data_dir.mkdir(exist_ok=True)
settings.processed_data_dir.mkdir(exist_ok=True)
settings.models_dir.mkdir(exist_ok=True)


# Conference mappings for standardization
CONFERENCE_MAPPINGS = {
    "Big Ten": "B1G",
    "SEC": "SEC", 
    "ACC": "ACC",
    "Big 12": "B12",
    "Pac-12": "P12",
    "American Athletic": "AAC",
    "Mountain West": "MWC",
    "MAC": "MAC",
    "Sun Belt": "SBC",
    "Conference USA": "CUSA",
    "Independent": "IND"
}

# Team statistics columns we want to collect
TEAM_STATS_COLUMNS = [
    "team",
    "conference", 
    "games",
    "wins",
    "losses",
    "ties",
    "points_for",
    "points_against",
    "yards_per_game",
    "yards_allowed_per_game",
    "turnovers",
    "turnover_margin",
    "red_zone_efficiency",
    "red_zone_defense",
    "third_down_conversion",
    "third_down_defense",
    "time_of_possession"
]

# Game statistics columns
GAME_STATS_COLUMNS = [
    "season",
    "week",
    "season_type",
    "start_date",
    "home_team",
    "away_team",
    "home_points",
    "away_points",
    "home_yards",
    "away_yards",
    "home_turnovers",
    "away_turnovers",
    "home_penalties",
    "away_penalties",
    "home_time_of_possession",
    "away_time_of_possession"
]


