"""
Data Transformation Module for NCAA Football Analytics

This module handles advanced data transformations, feature engineering,
and aggregation for machine learning and analytics purposes.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import warnings

from config.settings import settings


logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)


class DataTransformer:
    """Main data transformation class"""
    
    def __init__(self):
        """Initialize the data transformer"""
        self.processed_data_dir = settings.processed_data_dir
        self.models_dir = settings.models_dir
        
        # Ensure directories exist
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Data transformer initialized")
    
    def create_team_features(self, team_stats_df: pd.DataFrame, 
                           games_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create comprehensive team features for ML models
        
        Args:
            team_stats_df: Cleaned team statistics DataFrame
            games_df: Optional games DataFrame for additional features
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Creating team features...")
        
        features_df = team_stats_df.copy()
        
        # Basic performance features
        features_df = self._create_basic_features(features_df)
        
        # Advanced metrics
        features_df = self._create_advanced_metrics(features_df)
        
        # Conference-specific features
        features_df = self._create_conference_features(features_df)
        
        # Historical features (if multiple seasons available)
        features_df = self._create_historical_features(features_df)
        
        # Ensure we have a 'year' column for compatibility
        if 'season' in features_df.columns and 'year' not in features_df.columns:
            features_df['year'] = features_df['season']
        
        # Game-level features (if games data available)
        if games_df is not None:
            features_df = self._create_game_level_features(features_df, games_df)
        
        logger.info(f"Created features for {len(features_df)} teams")
        return features_df
    
    def create_prediction_dataset(self, features_df: pd.DataFrame, 
                                target_year: int = None) -> pd.DataFrame:
        """
        Create dataset for prediction models
        
        Args:
            features_df: Team features DataFrame
            target_year: Year to predict for (defaults to latest + 1)
            
        Returns:
            DataFrame ready for ML training/prediction
        """
        logger.info("Creating prediction dataset...")
        
        # Use 'season' if 'year' is not available
        year_col = 'year' if 'year' in features_df.columns else 'season'
        
        if target_year is None:
            target_year = features_df[year_col].max() + 1
        
        # Filter to relevant seasons (exclude target year if it exists)
        train_df = features_df[features_df[year_col] < target_year].copy()
        
        # Create target variables
        train_df = self._create_target_variables(train_df)
        
        # Select features for ML
        ml_df = self._select_ml_features(train_df)
        
        # Handle missing values and outliers
        ml_df = self._handle_ml_preprocessing(ml_df)
        
        logger.info(f"Created prediction dataset with {len(ml_df)} samples")
        return ml_df
    
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic performance features"""
        # Efficiency metrics (using yards instead of points)
        if all(col in df.columns for col in ['totalYards', 'games']):
            df['offensive_efficiency'] = df['totalYards'] / df['games'].replace(0, np.nan)
        
        if all(col in df.columns for col in ['totalYardsOpponent', 'games']):
            df['defensive_efficiency'] = df['totalYardsOpponent'] / df['games'].replace(0, np.nan)
        
        # Strength metrics (using win percentage)
        if 'win_percentage' in df.columns:
            df['win_strength'] = df['win_percentage']
        
        # Consistency metrics (if we have game-level data)
        if 'win_percentage' in df.columns:
            df['consistency_score'] = 1 - df['win_percentage'].std() if len(df) > 1 else 0.5
        
        return df
    
    def _create_advanced_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced performance metrics"""
        # Pythagorean expectation (using yards instead of points)
        if all(col in df.columns for col in ['totalYards', 'totalYardsOpponent']):
            df['pythagorean_expectation'] = df['totalYards'] ** 2.37 / (
                df['totalYards'] ** 2.37 + df['totalYardsOpponent'] ** 2.37
            )
        
        # Margin of victory metrics (using yards)
        if 'yard_differential' in df.columns and 'games' in df.columns:
            df['margin_of_victory'] = df['yard_differential'] / df['games'].replace(0, np.nan)
        
        # Turnover metrics (if available)
        if 'turnovers' in df.columns and 'games' in df.columns:
            df['turnovers_per_game'] = df['turnovers'] / df['games'].replace(0, np.nan)
        
        if 'turnover_margin' in df.columns and 'games' in df.columns:
            df['turnover_margin_per_game'] = df['turnover_margin'] / df['games'].replace(0, np.nan)
        
        # Offensive balance (rush vs pass)
        if all(col in df.columns for col in ['rushingYards', 'netPassingYards']):
            df['offensive_balance'] = df['rushingYards'] / (df['rushingYards'] + df['netPassingYards']).replace(0, np.nan)
        
        # Defensive pressure
        if all(col in df.columns for col in ['sacks', 'tacklesForLoss']):
            df['defensive_pressure'] = df['sacks'] + df['tacklesForLoss']
        
        return df
    
    def _create_conference_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create conference-specific features"""
        if 'conference' not in df.columns:
            return df
        
        # Use 'season' if 'year' is not available
        year_col = 'year' if 'year' in df.columns else 'season'
        
        # Conference strength (average win percentage)
        conference_strength = df.groupby('conference')['win_percentage'].mean()
        df['conference_strength'] = df['conference'].map(conference_strength)
        
        # Conference ranking within conference
        df['conference_rank'] = df.groupby([year_col, 'conference'])['win_percentage'].rank(
            method='dense', ascending=False
        )
        
        # Conference dominance (wins above conference average)
        df['conference_dominance'] = df['win_percentage'] - df['conference_strength']
        
        return df
    
    def _create_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create historical performance features"""
        # Use 'season' if 'year' is not available
        year_col = 'year' if 'year' in df.columns else 'season'
        
        if year_col not in df.columns or len(df[year_col].unique()) < 2:
            return df
        
        # Sort by team and year/season
        df = df.sort_values(['team', year_col])
        
        # Previous year performance
        df['prev_year_win_pct'] = df.groupby('team')['win_percentage'].shift(1)
        if 'yards_per_game' in df.columns:
            df['prev_year_yards_per_game'] = df.groupby('team')['yards_per_game'].shift(1)
        if 'yard_differential' in df.columns:
            df['prev_year_yard_differential'] = df.groupby('team')['yard_differential'].shift(1)
        
        # 3-year rolling averages
        df['win_pct_3yr'] = df.groupby('team')['win_percentage'].rolling(3, min_periods=1).mean().values
        if 'yards_per_game' in df.columns:
            df['yards_per_game_3yr'] = df.groupby('team')['yards_per_game'].rolling(3, min_periods=1).mean().values
        
        # Performance trend
        df['win_pct_trend'] = df.groupby('team')['win_percentage'].diff()
        
        return df
    
    def _create_game_level_features(self, team_df: pd.DataFrame, 
                                  games_df: pd.DataFrame) -> pd.DataFrame:
        """Create features from game-level data"""
        if games_df is None or len(games_df) == 0:
            return team_df
        
        # Calculate home/away performance
        home_stats = games_df.groupby('home_team').agg({
            'home_points': 'mean',
            'away_points': 'mean',
            'total_points': 'mean',
            'home_win': 'mean'
        }).add_prefix('home_')
        
        away_stats = games_df.groupby('away_team').agg({
            'home_points': 'mean',
            'away_points': 'mean',
            'total_points': 'mean',
            'away_win': 'mean'
        }).add_prefix('away_')
        
        # Merge with team data
        team_df = team_df.merge(home_stats, left_on='team', right_index=True, how='left')
        team_df = team_df.merge(away_stats, left_on='team', right_index=True, how='left')
        
        # Calculate home field advantage
        if 'home_home_win' in team_df.columns and 'away_away_win' in team_df.columns:
            team_df['home_field_advantage'] = (
                team_df['home_home_win'] - team_df['away_away_win']
            )
        
        return team_df
    
    def _create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for ML models"""
        # National champion target (binary)
        df['is_national_champion'] = 0
        
        # Conference champion target (multi-class)
        df['conference_champion'] = 'None'
        
        # Use 'season' if 'year' is not available
        year_col = 'year' if 'year' in df.columns else 'season'
        
        # For each year, identify champions
        for year in df[year_col].unique():
            year_data = df[df[year_col] == year]
            
            # National champion (highest win percentage or specific criteria)
            if len(year_data) > 0:
                national_champ_idx = year_data['win_percentage'].idxmax()
                df.loc[national_champ_idx, 'is_national_champion'] = 1
            
            # Conference champions
            for conference in year_data['conference'].unique():
                conf_data = year_data[year_data['conference'] == conference]
                if len(conf_data) > 0:
                    conf_champ_idx = conf_data['win_percentage'].idxmax()
                    df.loc[conf_champ_idx, 'conference_champion'] = conference
        
        return df
    
    def _select_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select features for machine learning"""
        # Define feature columns (exclude identifiers and targets)
        exclude_columns = [
            'team', 'conference', 'year', 'season', 'start_date',
            'is_national_champion', 'conference_champion'
        ]
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Select numeric features only
        numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        
        # Create ML dataset
        ml_df = df[['team', 'year', 'conference', 'is_national_champion', 'conference_champion'] + 
                  numeric_features].copy()
        
        logger.info(f"Selected {len(numeric_features)} features for ML")
        return ml_df
    
    def _handle_ml_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle preprocessing for ML models"""
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values with median for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isna().any():
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
        
        # Remove rows with too many missing values
        threshold = 0.5  # Remove rows with >50% missing values
        df = df.dropna(thresh=int(len(df.columns) * (1 - threshold)))
        
        return df
    
    def create_time_series_features(self, df: pd.DataFrame, 
                                  target_columns: List[str] = None) -> pd.DataFrame:
        """
        Create time series features for trend analysis
        
        Args:
            df: DataFrame with time series data
            target_columns: Columns to create time series features for
            
        Returns:
            DataFrame with time series features
        """
        if target_columns is None:
            target_columns = ['win_percentage', 'points_per_game', 'point_differential']
        
        ts_df = df.copy()
        ts_df = ts_df.sort_values(['team', 'year'])
        
        for col in target_columns:
            if col in ts_df.columns:
                # Rolling averages
                ts_df[f'{col}_ma_3'] = ts_df.groupby('team')[col].rolling(3, min_periods=1).mean().values
                ts_df[f'{col}_ma_5'] = ts_df.groupby('team')[col].rolling(5, min_periods=1).mean().values
                
                # Trends
                ts_df[f'{col}_trend'] = ts_df.groupby('team')[col].diff()
                ts_df[f'{col}_trend_2'] = ts_df.groupby('team')[col].diff(2)
                
                # Volatility
                ts_df[f'{col}_volatility'] = ts_df.groupby('team')[col].rolling(3).std().values
        
        return ts_df
    
    def aggregate_by_conference(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate team data by conference"""
        if 'conference' not in df.columns:
            return df
        
        agg_dict = {
            'win_percentage': 'mean',
            'points_per_game': 'mean',
            'points_against_per_game': 'mean',
            'point_differential': 'mean',
            'games': 'sum',
            'wins': 'sum',
            'losses': 'sum'
        }
        
        # Only include columns that exist
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
        
        conference_df = df.groupby(['conference', 'year']).agg(agg_dict).reset_index()
        
        return conference_df
    
    def export_for_ml(self, df: pd.DataFrame, filename: str = None) -> str:
        """Export processed data for ML models"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ml_dataset_{timestamp}.csv"
        
        output_path = self.models_dir / filename
        df.to_csv(output_path, index=False)
        
        logger.info(f"Exported ML dataset to {output_path}")
        return str(output_path)


def main():
    """Main function for running data transformation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Transform NCAA Football Data")
    parser.add_argument("--input-file", required=True, help="Input CSV file")
    parser.add_argument("--output-file", help="Output CSV file")
    parser.add_argument("--target-year", type=int, help="Target year for prediction")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load data
    df = pd.read_csv(args.input_file)
    
    # Create transformer and process
    transformer = DataTransformer()
    
    # Create features
    features_df = transformer.create_team_features(df)
    
    # Create prediction dataset
    ml_df = transformer.create_prediction_dataset(features_df, args.target_year)
    
    # Export
    output_file = args.output_file or "transformed_data.csv"
    transformer.export_for_ml(ml_df, output_file)
    
    print(f"Transformation completed. Output saved to {output_file}")


if __name__ == "__main__":
    main()


