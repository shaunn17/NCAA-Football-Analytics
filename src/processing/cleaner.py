"""
Data Cleaning Module for NCAA Football Analytics

This module handles cleaning and standardizing data from the College Football Data API,
including handling missing values, standardizing team names, and creating derived metrics.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import re

from config.settings import settings, CONFERENCE_MAPPINGS


logger = logging.getLogger(__name__)


class DataCleaner:
    """Main data cleaning class"""
    
    def __init__(self):
        """Initialize the data cleaner"""
        self.raw_data_dir = settings.raw_data_dir
        self.processed_data_dir = settings.processed_data_dir
        self.conference_mappings = CONFERENCE_MAPPINGS
        
        # Ensure processed data directory exists
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Data cleaner initialized")
    
    def clean_team_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean team statistics data
        
        Args:
            df: Raw team stats DataFrame
            
        Returns:
            Cleaned team stats DataFrame
        """
        logger.info(f"Cleaning team stats data with {len(df)} rows")
        
        # Create a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Standardize column names
        cleaned_df.columns = [self._standardize_column_name(col) for col in cleaned_df.columns]
        
        # Check if data is in long format (statname/statvalue) and pivot if needed
        if 'statname' in cleaned_df.columns and 'statvalue' in cleaned_df.columns:
            logger.info("Converting long format to wide format")
            # Pivot the data to wide format
            cleaned_df = cleaned_df.pivot_table(
                index=['season', 'team', 'conference'], 
                columns='statname', 
                values='statvalue', 
                aggfunc='first'
            ).reset_index()
            
            # Flatten column names
            cleaned_df.columns.name = None
        
        # Clean team names
        if 'team' in cleaned_df.columns:
            cleaned_df['team'] = cleaned_df['team'].apply(self._clean_team_name)
        elif 'school' in cleaned_df.columns:
            cleaned_df['team'] = cleaned_df['school'].apply(self._clean_team_name)
        
        # Clean conference names
        if 'conference' in cleaned_df.columns:
            cleaned_df['conference'] = cleaned_df['conference'].apply(self._standardize_conference)
        
        # Handle numeric columns
        numeric_columns = ['games', 'wins', 'losses', 'ties', 'points_for', 'points_against']
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        
        # Calculate derived metrics
        cleaned_df = self._calculate_derived_metrics(cleaned_df)
        
        # Handle missing values
        cleaned_df = self._handle_missing_values(cleaned_df)
        
        # Remove duplicates (only use year if it exists)
        if 'year' in cleaned_df.columns:
            cleaned_df = cleaned_df.drop_duplicates(subset=['team', 'year'], keep='last')
        elif 'season' in cleaned_df.columns:
            cleaned_df = cleaned_df.drop_duplicates(subset=['team', 'season'], keep='last')
        else:
            cleaned_df = cleaned_df.drop_duplicates(subset=['team'], keep='last')
        
        logger.info(f"Cleaned team stats data: {len(cleaned_df)} rows")
        return cleaned_df
    
    def clean_games(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean games data
        
        Args:
            df: Raw games DataFrame
            
        Returns:
            Cleaned games DataFrame
        """
        logger.info(f"Cleaning games data with {len(df)} rows")
        
        cleaned_df = df.copy()
        
        # Standardize column names
        cleaned_df.columns = [self._standardize_column_name(col) for col in cleaned_df.columns]
        
        # Clean team names
        team_columns = ['home_team', 'away_team']
        for col in team_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].apply(self._clean_team_name)
        
        # Handle date columns
        if 'start_date' in cleaned_df.columns:
            cleaned_df['start_date'] = pd.to_datetime(cleaned_df['start_date'], errors='coerce')
            cleaned_df['year'] = cleaned_df['start_date'].dt.year
            cleaned_df['month'] = cleaned_df['start_date'].dt.month
            cleaned_df['day'] = cleaned_df['start_date'].dt.day
            cleaned_df['weekday'] = cleaned_df['start_date'].dt.day_name()
        
        # Handle numeric columns
        numeric_columns = ['week', 'home_points', 'away_points', 'home_yards', 'away_yards']
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        
        # Calculate game metrics
        cleaned_df = self._calculate_game_metrics(cleaned_df)
        
        # Handle missing values
        cleaned_df = self._handle_missing_values(cleaned_df)
        
        logger.info(f"Cleaned games data: {len(cleaned_df)} rows")
        return cleaned_df
    
    def clean_rankings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean rankings data
        
        Args:
            df: Raw rankings DataFrame
            
        Returns:
            Cleaned rankings DataFrame
        """
        logger.info(f"Cleaning rankings data with {len(df)} rows")
        
        cleaned_df = df.copy()
        
        # Standardize column names
        cleaned_df.columns = [self._standardize_column_name(col) for col in cleaned_df.columns]
        
        # Clean team names
        if 'team' in cleaned_df.columns:
            cleaned_df['team'] = cleaned_df['team'].apply(self._clean_team_name)
        
        # Handle numeric columns
        numeric_columns = ['year', 'week', 'rank']
        for col in numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        
        # Handle missing values
        cleaned_df = self._handle_missing_values(cleaned_df)
        
        logger.info(f"Cleaned rankings data: {len(cleaned_df)} rows")
        return cleaned_df
    
    def clean_conferences(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean conferences data
        
        Args:
            df: Raw conferences DataFrame
            
        Returns:
            Cleaned conferences DataFrame
        """
        logger.info(f"Cleaning conferences data with {len(df)} rows")
        
        cleaned_df = df.copy()
        
        # Standardize column names
        cleaned_df.columns = [self._standardize_column_name(col) for col in cleaned_df.columns]
        
        # Clean conference names
        if 'name' in cleaned_df.columns:
            cleaned_df['standardized_name'] = cleaned_df['name'].apply(self._standardize_conference)
        
        # Add conference mappings
        cleaned_df['conference_code'] = cleaned_df['standardized_name'].map(
            {v: k for k, v in self.conference_mappings.items()}
        )
        
        logger.info(f"Cleaned conferences data: {len(cleaned_df)} rows")
        return cleaned_df
    
    def _standardize_column_name(self, col_name: str) -> str:
        """Standardize column names"""
        # Convert to lowercase and replace spaces/special chars with underscores
        standardized = re.sub(r'[^a-zA-Z0-9]', '_', str(col_name).lower())
        # Remove multiple underscores
        standardized = re.sub(r'_+', '_', standardized)
        # Remove leading/trailing underscores
        standardized = standardized.strip('_')
        
        return standardized
    
    def _clean_team_name(self, team_name: str) -> str:
        """Clean and standardize team names"""
        if pd.isna(team_name):
            return team_name
        
        team_name = str(team_name).strip()
        
        # Common team name standardizations
        team_mappings = {
            'Ohio State': 'Ohio State',
            'Ohio St.': 'Ohio State',
            'Ohio St': 'Ohio State',
            'Michigan State': 'Michigan State',
            'Michigan St.': 'Michigan State',
            'Michigan St': 'Michigan State',
            'Penn State': 'Penn State',
            'Penn St.': 'Penn State',
            'Penn St': 'Penn State',
            'Florida State': 'Florida State',
            'Florida St.': 'Florida State',
            'Florida St': 'Florida State',
            'Virginia Tech': 'Virginia Tech',
            'Virginia Polytechnic Institute': 'Virginia Tech',
            'Texas A&M': 'Texas A&M',
            'Texas A & M': 'Texas A&M',
            'Miami (FL)': 'Miami (FL)',
            'Miami': 'Miami (FL)',
            'Miami (OH)': 'Miami (OH)',
            'Miami (Ohio)': 'Miami (OH)',
            'UCF': 'UCF',
            'Central Florida': 'UCF',
            'USC': 'USC',
            'Southern California': 'USC',
            'UCLA': 'UCLA',
            'California-Los Angeles': 'UCLA',
            'LSU': 'LSU',
            'Louisiana State': 'LSU',
            'Ole Miss': 'Ole Miss',
            'Mississippi': 'Ole Miss',
            'Mississippi State': 'Mississippi State',
            'Mississippi St.': 'Mississippi State',
            'Mississippi St': 'Mississippi State',
        }
        
        return team_mappings.get(team_name, team_name)
    
    def _standardize_conference(self, conference_name: str) -> str:
        """Standardize conference names"""
        if pd.isna(conference_name):
            return conference_name
        
        conference_name = str(conference_name).strip()
        
        # Map to standardized names
        return self.conference_mappings.get(conference_name, conference_name)
    
    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived metrics for team stats"""
        # Win percentage (if wins and games are available)
        if all(col in df.columns for col in ['wins', 'games']):
            df['win_percentage'] = df['wins'] / df['games'].replace(0, np.nan)
        elif 'games' in df.columns:
            # If we don't have wins, we can't calculate win percentage
            df['win_percentage'] = np.nan
        
        # Yards per game (offense and defense)
        if all(col in df.columns for col in ['totalYards', 'games']):
            df['yards_per_game'] = df['totalYards'] / df['games'].replace(0, np.nan)
        
        if all(col in df.columns for col in ['totalYardsOpponent', 'games']):
            df['yards_allowed_per_game'] = df['totalYardsOpponent'] / df['games'].replace(0, np.nan)
        
        # Yard differential
        if all(col in df.columns for col in ['totalYards', 'totalYardsOpponent']):
            df['yard_differential'] = df['totalYards'] - df['totalYardsOpponent']
        
        # Turnover margin
        if all(col in df.columns for col in ['turnovers', 'turnoversOpponent']):
            df['turnover_margin'] = df['turnoversOpponent'] - df['turnovers']
        
        # First down efficiency
        if all(col in df.columns for col in ['firstDowns', 'firstDownsOpponent']):
            df['first_down_differential'] = df['firstDowns'] - df['firstDownsOpponent']
        
        # Third down conversion rate (if available)
        if all(col in df.columns for col in ['thirdDownConversions', 'thirdDowns']):
            df['third_down_conversion_rate'] = df['thirdDownConversions'] / df['thirdDowns'].replace(0, np.nan)
        
        # Fourth down conversion rate (if available)
        if all(col in df.columns for col in ['fourthDownConversions', 'fourthDowns']):
            df['fourth_down_conversion_rate'] = df['fourthDownConversions'] / df['fourthDowns'].replace(0, np.nan)
        
        # Rushing efficiency
        if all(col in df.columns for col in ['rushingYards', 'rushingAttempts']):
            df['rushing_yards_per_attempt'] = df['rushingYards'] / df['rushingAttempts'].replace(0, np.nan)
        
        # Passing efficiency
        if all(col in df.columns for col in ['netPassingYards', 'passAttempts']):
            df['passing_yards_per_attempt'] = df['netPassingYards'] / df['passAttempts'].replace(0, np.nan)
        
        # Completion percentage
        if all(col in df.columns for col in ['passCompletions', 'passAttempts']):
            df['completion_percentage'] = df['passCompletions'] / df['passAttempts'].replace(0, np.nan)
        
        return df
    
    def _calculate_game_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived metrics for games"""
        # Point differential
        if all(col in df.columns for col in ['home_points', 'away_points']):
            df['point_differential'] = df['home_points'] - df['away_points']
            df['home_win'] = df['point_differential'] > 0
            df['away_win'] = df['point_differential'] < 0
            df['tie'] = df['point_differential'] == 0
        
        # Total points
        if all(col in df.columns for col in ['home_points', 'away_points']):
            df['total_points'] = df['home_points'] + df['away_points']
        
        # Total yards (if available)
        if all(col in df.columns for col in ['home_yards', 'away_yards']):
            df['total_yards'] = df['home_yards'] + df['away_yards']
            df['yard_differential'] = df['home_yards'] - df['away_yards']
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # For numeric columns, fill with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isna().any():
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
        
        # For categorical columns, fill with mode or 'Unknown'
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isna().any():
                mode_value = df[col].mode()
                fill_value = mode_value[0] if len(mode_value) > 0 else 'Unknown'
                df[col] = df[col].fillna(fill_value)
        
        return df
    
    def process_all_data(self, seasons: List[int] = None) -> Dict[str, Any]:
        """
        Process all raw data files
        
        Args:
            seasons: List of seasons to process
            
        Returns:
            Dictionary with processing results
        """
        seasons = seasons or settings.seasons_to_collect
        
        results = {
            "start_time": pd.Timestamp.now().isoformat(),
            "seasons": seasons,
            "processed_files": {},
            "errors": []
        }
        
        try:
            # Process conferences
            logger.info("Processing conferences data...")
            conf_file = self.raw_data_dir / "conferences.csv"
            if conf_file.exists():
                df = pd.read_csv(conf_file)
                cleaned_df = self.clean_conferences(df)
                output_file = self.processed_data_dir / "conferences_clean.csv"
                cleaned_df.to_csv(output_file, index=False)
                results["processed_files"]["conferences"] = str(output_file)
            
            # Process teams
            logger.info("Processing teams data...")
            teams_file = self.raw_data_dir / "teams_all.csv"
            if teams_file.exists():
                df = pd.read_csv(teams_file)
                cleaned_df = self.clean_team_stats(df)
                output_file = self.processed_data_dir / "teams_clean.csv"
                cleaned_df.to_csv(output_file, index=False)
                results["processed_files"]["teams"] = str(output_file)
            
            # Process season-specific data
            for season in seasons:
                logger.info(f"Processing data for season {season}...")
                season_results = self._process_season_data(season)
                results["processed_files"][f"season_{season}"] = season_results
            
            results["end_time"] = pd.Timestamp.now().isoformat()
            results["success"] = True
            
            logger.info("Data processing completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            results["error"] = str(e)
            results["success"] = False
            results["end_time"] = pd.Timestamp.now().isoformat()
            return results
    
    def _process_season_data(self, season: int) -> Dict[str, str]:
        """Process data for a specific season"""
        season_results = {}
        
        # Process team stats
        stats_file = self.raw_data_dir / f"team_stats_{season}.csv"
        if stats_file.exists():
            df = pd.read_csv(stats_file)
            cleaned_df = self.clean_team_stats(df)
            output_file = self.processed_data_dir / f"team_stats_{season}_clean.csv"
            cleaned_df.to_csv(output_file, index=False)
            season_results["team_stats"] = str(output_file)
        
        # Process games
        games_file = self.raw_data_dir / f"games_{season}.csv"
        if games_file.exists():
            df = pd.read_csv(games_file)
            cleaned_df = self.clean_games(df)
            output_file = self.processed_data_dir / f"games_{season}_clean.csv"
            cleaned_df.to_csv(output_file, index=False)
            season_results["games"] = str(output_file)
        
        # Process rankings
        rankings_file = self.raw_data_dir / f"rankings_{season}.csv"
        if rankings_file.exists():
            df = pd.read_csv(rankings_file)
            cleaned_df = self.clean_rankings(df)
            output_file = self.processed_data_dir / f"rankings_{season}_clean.csv"
            cleaned_df.to_csv(output_file, index=False)
            season_results["rankings"] = str(output_file)
        
        return season_results


def main():
    """Main function for running data cleaning"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean NCAA Football Data")
    parser.add_argument("--seasons", nargs="+", type=int, 
                       help="Seasons to process (default: from settings)")
    parser.add_argument("--input-dir", help="Input directory (default: raw data dir)")
    parser.add_argument("--output-dir", help="Output directory (default: processed data dir)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create cleaner and run
    cleaner = DataCleaner()
    results = cleaner.process_all_data(args.seasons)
    
    print(f"Processing completed: {results['success']}")
    if not results['success']:
        print(f"Error: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()


