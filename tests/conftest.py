"""
Test configuration and fixtures for NCAA Football Analytics

This module provides shared test fixtures and configuration for all tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys
from unittest.mock import Mock, patch

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture
def sample_team_data():
    """Sample team data for testing"""
    return pd.DataFrame({
        'team': ['Indiana', 'Ohio State', 'Michigan', 'Penn State'],
        'year': [2024, 2024, 2024, 2024],
        'conference': ['B1G', 'B1G', 'B1G', 'B1G'],
        'wins': [11, 14, 12, 10],
        'losses': [2, 1, 2, 3],
        'games': [13, 15, 14, 13],
        'win_percentage': [0.846, 0.933, 0.857, 0.769],
        'totalYards': [5558, 6873, 6000, 5200],
        'totalYardsOpponent': [3332, 3822, 3500, 3900],
        'yards_per_game': [427.5, 458.2, 428.6, 400.0],
        'yards_allowed_per_game': [256.3, 254.8, 250.0, 300.0],
        'turnover_margin': [15, 5, 8, 2]
    })

@pytest.fixture
def sample_games_data():
    """Sample games data for testing"""
    return pd.DataFrame({
        'homeTeam': ['Indiana', 'Ohio State', 'Michigan'],
        'awayTeam': ['Purdue', 'Michigan', 'Ohio State'],
        'homePoints': [35, 42, 28],
        'awayPoints': [14, 24, 30],
        'season': [2024, 2024, 2024],
        'completed': [True, True, True]
    })

@pytest.fixture
def sample_predictions():
    """Sample ML predictions for testing"""
    return pd.DataFrame({
        'team': ['Indiana', 'Ohio State', 'Michigan', 'Penn State'],
        'top_25_probability': [0.85, 0.95, 0.90, 0.75],
        'performance_rank': [12, 3, 8, 18]
    })

@pytest.fixture
def temp_data_dir():
    """Temporary directory for test data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def mock_api_response():
    """Mock API response for testing"""
    return {
        'teams': [
            {'id': 1, 'school': 'Indiana', 'conference': 'B1G'},
            {'id': 2, 'school': 'Ohio State', 'conference': 'B1G'}
        ],
        'games': [
            {
                'id': 1,
                'homeTeam': 'Indiana',
                'awayTeam': 'Purdue',
                'homePoints': 35,
                'awayPoints': 14,
                'season': 2024,
                'completed': True
            }
        ]
    }

@pytest.fixture
def mock_database():
    """Mock database for testing"""
    mock_db = Mock()
    mock_db.query.return_value = pd.DataFrame({
        'team': ['Indiana', 'Ohio State'],
        'win_percentage': [0.846, 0.933]
    })
    mock_db.get_database_stats.return_value = {
        'total_records': 268,
        'unique_teams': 134,
        'seasons': 2,
        'conferences': 11
    }
    return mock_db

@pytest.fixture
def test_env_vars():
    """Set up test environment variables"""
    original_env = os.environ.copy()
    os.environ['CFBD_API_KEY'] = 'test_api_key'
    yield
    os.environ.clear()
    os.environ.update(original_env)

# Test data validation helpers
def assert_dataframe_not_empty(df, name="DataFrame"):
    """Assert that a DataFrame is not empty"""
    assert not df.empty, f"{name} should not be empty"

def assert_column_exists(df, column, name="DataFrame"):
    """Assert that a column exists in DataFrame"""
    assert column in df.columns, f"{name} should have column '{column}'"

def assert_numeric_range(values, min_val=0, max_val=1, name="values"):
    """Assert that values are within a numeric range"""
    assert all(min_val <= val <= max_val for val in values), \
        f"{name} should be between {min_val} and {max_val}"

def assert_no_missing_values(df, critical_columns, name="DataFrame"):
    """Assert that critical columns have no missing values"""
    for col in critical_columns:
        if col in df.columns:
            assert not df[col].isna().any(), f"{name} should have no missing values in column '{col}'"

# Performance testing helpers
def assert_response_time(func, max_seconds=5):
    """Assert that a function executes within time limit"""
    import time
    start_time = time.time()
    result = func()
    execution_time = time.time() - start_time
    assert execution_time < max_seconds, f"Function took {execution_time:.2f}s, should be < {max_seconds}s"
    return result
