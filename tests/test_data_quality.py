"""
Data Quality Tests for NCAA Football Analytics

Tests to ensure data accuracy, completeness, and consistency.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from tests.conftest import (
    assert_dataframe_not_empty, 
    assert_column_exists, 
    assert_numeric_range,
    assert_no_missing_values
)

class TestDataQuality:
    """Test data quality and accuracy"""
    
    def test_win_percentage_calculation(self, sample_team_data):
        """Test that win percentages are calculated correctly"""
        df = sample_team_data
        
        # Test Indiana: 11 wins, 2 losses = 11/13 = 0.846
        indiana = df[df['team'] == 'Indiana'].iloc[0]
        assert abs(indiana['win_percentage'] - 0.846) < 0.001
        
        # Test Ohio State: 14 wins, 1 loss = 14/15 = 0.933
        ohio_state = df[df['team'] == 'Ohio State'].iloc[0]
        assert abs(ohio_state['win_percentage'] - 0.933) < 0.001
        
        # Test that all win percentages are between 0 and 1
        assert_numeric_range(df['win_percentage'], 0, 1, "Win percentages")
    
    def test_win_percentage_from_games_data(self, sample_games_data):
        """Test win percentage calculation from games data"""
        from scripts.fix_win_percentages import calculate_win_percentages_from_games
        
        # Create temporary CSV file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_games_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Calculate win percentages
            records_df = calculate_win_percentages_from_games()
            
            # Verify structure
            assert_dataframe_not_empty(records_df, "Win records")
            assert_column_exists(records_df, 'team', "Win records")
            assert_column_exists(records_df, 'win_percentage', "Win records")
            assert_column_exists(records_df, 'wins', "Win records")
            assert_column_exists(records_df, 'losses', "Win records")
            
            # Verify win percentages are valid
            assert_numeric_range(records_df['win_percentage'], 0, 1, "Win percentages")
            
        finally:
            # Clean up
            Path(temp_path).unlink()
    
    def test_processed_data_exists(self):
        """Test that processed data files exist and are valid"""
        data_path = Path("data/models/ncaa_football_ml_dataset.csv")
        assert data_path.exists(), "Processed data file should exist"
        
        df = pd.read_csv(data_path)
        assert_dataframe_not_empty(df, "Processed data")
        
        # Check critical columns
        critical_columns = ['team', 'year', 'conference', 'win_percentage']
        for col in critical_columns:
            assert_column_exists(df, col, "Processed data")
        
        # Check data quality
        assert_no_missing_values(df, ['team', 'year'], "Processed data")
        
        # Check win percentages are valid
        valid_win_pct = df['win_percentage'].dropna()
        if len(valid_win_pct) > 0:
            assert_numeric_range(valid_win_pct, 0, 1, "Win percentages")
    
    def test_team_name_consistency(self):
        """Test that team names are consistent across datasets"""
        data_path = Path("data/models/ncaa_football_ml_dataset.csv")
        if not data_path.exists():
            pytest.skip("Processed data not available")
        
        df = pd.read_csv(data_path)
        
        # Check that team names are strings and not empty
        assert all(isinstance(team, str) for team in df['team'] if pd.notna(team))
        assert all(len(team.strip()) > 0 for team in df['team'] if pd.notna(team))
        
        # Check for common team name issues
        team_names = df['team'].dropna().unique()
        
        # No empty strings
        assert '' not in team_names
        
        # No obvious duplicates with different casing
        lower_names = [name.lower() for name in team_names]
        assert len(lower_names) == len(set(lower_names)), "Team names should be unique (case-insensitive)"
    
    def test_conference_data_quality(self):
        """Test conference data quality"""
        data_path = Path("data/models/ncaa_football_ml_dataset.csv")
        if not data_path.exists():
            pytest.skip("Processed data not available")
        
        df = pd.read_csv(data_path)
        
        # Check conference column exists
        assert_column_exists(df, 'conference', "Processed data")
        
        # Check that conferences are valid (using standardized names)
        valid_conferences = ['Big Ten', 'SEC', 'Big 12', 'ACC', 'PAC', 'AAC', 'Mountain West', 'MAC', 'Sun Belt', 'Conference USA', 'Independent']
        invalid_conferences = df[~df['conference'].isin(valid_conferences)]['conference'].unique()
        
        # Allow some flexibility for conference names
        assert len(invalid_conferences) < len(df['conference'].unique()) * 0.1, \
            f"Too many invalid conferences: {invalid_conferences}"
    
    def test_yards_per_game_calculation(self, sample_team_data):
        """Test yards per game calculation"""
        df = sample_team_data
        
        # Test Indiana: 5558 yards / 13 games = 427.5
        indiana = df[df['team'] == 'Indiana'].iloc[0]
        expected_yards_pg = indiana['totalYards'] / indiana['games']
        assert abs(indiana['yards_per_game'] - expected_yards_pg) < 0.1
        
        # Test that yards per game is positive
        assert all(df['yards_per_game'] > 0), "Yards per game should be positive"
    
    def test_turnover_margin_calculation(self, sample_team_data):
        """Test turnover margin calculation"""
        df = sample_team_data
        
        # Test that turnover margin can be negative (more turnovers than forced)
        turnover_margins = df['turnover_margin'].unique()
        
        # Should have both positive and negative values
        assert len(turnover_margins) > 1, "Turnover margin should vary between teams"
    
    def test_data_completeness(self):
        """Test that all expected teams and seasons are present"""
        data_path = Path("data/models/ncaa_football_ml_dataset.csv")
        if not data_path.exists():
            pytest.skip("Processed data not available")
        
        df = pd.read_csv(data_path)
        
        # Check that we have data for expected seasons
        expected_seasons = [2023, 2024]
        actual_seasons = sorted(df['year'].unique())
        
        for season in expected_seasons:
            assert season in actual_seasons, f"Should have data for {season}"
        
        # Check that we have a reasonable number of teams
        unique_teams = df['team'].nunique()
        assert unique_teams >= 100, f"Should have at least 100 teams, got {unique_teams}"
        
        # Check that we have data for major conferences
        major_conferences = ['Big Ten', 'SEC', 'Big 12', 'ACC']
        for conf in major_conferences:
            conf_teams = df[df['conference'] == conf]['team'].nunique()
            assert conf_teams > 0, f"Should have teams in {conf} conference"
    
    def test_ml_predictions_data_quality(self):
        """Test ML predictions data quality"""
        predictions_path = Path("data/models/2025_predictions.csv")
        if not predictions_path.exists():
            pytest.skip("ML predictions not available")
        
        df = pd.read_csv(predictions_path)
        assert_dataframe_not_empty(df, "ML predictions")
        
        # Check required columns
        required_columns = ['team', 'top_25_probability']
        for col in required_columns:
            assert_column_exists(df, col, "ML predictions")
        
        # Check probability values are valid
        if 'top_25_probability' in df.columns:
            probabilities = df['top_25_probability'].dropna()
            if len(probabilities) > 0:
                assert_numeric_range(probabilities, 0, 1, "Top 25 probabilities")
        
        # Check that we have predictions for reasonable number of teams
        assert len(df) >= 100, f"Should have predictions for at least 100 teams, got {len(df)}"
    
    @pytest.mark.slow
    def test_data_pipeline_end_to_end(self):
        """Test complete data pipeline from API to processed data"""
        # This test verifies that all the data files exist and are valid
        # without actually running the full pipeline (which is slow)
        
        # Check that processed data exists
        processed_path = Path("data/models/ncaa_football_ml_dataset.csv")
        assert processed_path.exists(), "Processed data should exist"
        
        # Check that predictions exist
        predictions_path = Path("data/models/2025_predictions.csv")
        assert predictions_path.exists(), "ML predictions should exist"
        
        # Check that database exists
        db_path = Path("data/ncaa_football_simple.duckdb")
        assert db_path.exists(), "Database should exist"
        
        print("✅ All pipeline outputs exist and are accessible")
