"""
Database Tests for NCAA Football Analytics

Tests to ensure database operations work correctly and maintain data integrity.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from tests.conftest import assert_dataframe_not_empty, assert_column_exists, assert_response_time

class TestDatabaseOperations:
    """Test database operations and data integrity"""
    
    def test_database_file_exists(self):
        """Test that database file exists"""
        db_path = Path("data/ncaa_football_simple.duckdb")
        assert db_path.exists(), "Database file should exist"
    
    def test_database_connection(self):
        """Test database connection"""
        try:
            from src.storage.simple_database import create_simple_database
            db = create_simple_database()
            assert db is not None, "Database connection should be successful"
            db.close()
        except ImportError:
            pytest.skip("Database module not available")
        except Exception as e:
            pytest.fail(f"Database connection failed: {e}")
    
    def test_database_schema(self):
        """Test database schema"""
        try:
            from src.storage.simple_database import create_simple_database
            db = create_simple_database()
            
            # Test that main table exists
            result = db.query("SELECT name FROM sqlite_master WHERE type='table' AND name='ncaa_football_data'")
            assert len(result) > 0, "Main table should exist"
            
            # Test table structure
            columns = db.query("PRAGMA table_info(ncaa_football_data)")
            assert len(columns) > 0, "Table should have columns"
            
            db.close()
        except ImportError:
            pytest.skip("Database module not available")
        except Exception as e:
            pytest.fail(f"Database schema test failed: {e}")
    
    def test_database_data_loading(self):
        """Test that data is loaded in database"""
        try:
            from src.storage.simple_database import create_simple_database
            db = create_simple_database()
            
            # Test data count
            result = db.query("SELECT COUNT(*) as count FROM ncaa_football_data")
            count = result.iloc[0]['count']
            assert count > 0, f"Database should have data, got {count} records"
            
            # Test data structure
            sample_data = db.query("SELECT * FROM ncaa_football_data LIMIT 5")
            assert_dataframe_not_empty(sample_data, "Database sample data")
            
            # Check required columns
            required_columns = ['team', 'year', 'conference']
            for col in required_columns:
                assert_column_exists(sample_data, col, "Database data")
            
            db.close()
        except ImportError:
            pytest.skip("Database module not available")
        except Exception as e:
            pytest.fail(f"Database data loading test failed: {e}")
    
    def test_database_queries(self):
        """Test database query functions"""
        try:
            from src.storage.simple_database import create_simple_database
            db = create_simple_database()
            
            # Test get_team_stats
            team_stats = db.get_team_stats('Indiana', 2024)
            if not team_stats.empty:
                assert_column_exists(team_stats, 'team', "Team stats")
                assert_column_exists(team_stats, 'year', "Team stats")
            
            # Test get_top_teams
            top_teams = db.get_top_teams(2024, 10)
            assert_dataframe_not_empty(top_teams, "Top teams")
            assert_column_exists(top_teams, 'team', "Top teams")
            assert_column_exists(top_teams, 'win_percentage', "Top teams")
            
            # Test get_big_ten_teams
            big_ten = db.get_big_ten_teams(2024)
            if not big_ten.empty:
                assert_column_exists(big_ten, 'team', "Big Ten teams")
                assert_column_exists(big_ten, 'conference', "Big Ten teams")
            
            db.close()
        except ImportError:
            pytest.skip("Database module not available")
        except Exception as e:
            pytest.fail(f"Database queries test failed: {e}")
    
    def test_database_performance(self):
        """Test database query performance"""
        try:
            from src.storage.simple_database import create_simple_database
            db = create_simple_database()
            
            # Test query performance
            def query_teams():
                return db.get_top_teams(2024, 25)
            
            # Should respond within 2 seconds
            result = assert_response_time(query_teams, max_seconds=2)
            assert_dataframe_not_empty(result, "Performance test result")
            
            db.close()
        except ImportError:
            pytest.skip("Database module not available")
        except Exception as e:
            pytest.fail(f"Database performance test failed: {e}")
    
    def test_database_data_integrity(self):
        """Test database data integrity"""
        try:
            from src.storage.simple_database import create_simple_database
            db = create_simple_database()
            
            # Test data consistency
            all_data = db.query("SELECT * FROM ncaa_football_data")
            
            # Check for duplicate records
            duplicates = all_data.duplicated(subset=['team', 'year']).sum()
            assert duplicates == 0, f"Should have no duplicate team-year combinations, found {duplicates}"
            
            # Check data types
            if 'win_percentage' in all_data.columns:
                win_pct = all_data['win_percentage'].dropna()
                if len(win_pct) > 0:
                    assert all(0 <= pct <= 1 for pct in win_pct), "Win percentages should be between 0 and 1"
            
            # Check year range
            if 'year' in all_data.columns:
                years = all_data['year'].dropna()
                assert all(2020 <= year <= 2025 for year in years), "Years should be reasonable"
            
            db.close()
        except ImportError:
            pytest.skip("Database module not available")
        except Exception as e:
            pytest.fail(f"Database data integrity test failed: {e}")
    
    def test_database_indexes(self):
        """Test database indexes for performance"""
        try:
            from src.storage.simple_database import create_simple_database
            db = create_simple_database()
            
            # Check that indexes exist
            indexes = db.query("SELECT name FROM sqlite_master WHERE type='index'")
            index_names = indexes['name'].tolist()
            
            # Should have indexes for common queries
            expected_indexes = ['idx_team_year', 'idx_conference_year', 'idx_year']
            for idx in expected_indexes:
                assert idx in index_names, f"Should have index {idx}"
            
            db.close()
        except ImportError:
            pytest.skip("Database module not available")
        except Exception as e:
            pytest.fail(f"Database indexes test failed: {e}")
    
    def test_database_error_handling(self):
        """Test database error handling"""
        try:
            from src.storage.simple_database import create_simple_database
            db = create_simple_database()
            
            # Test invalid query
            with pytest.raises(Exception):
                db.query("SELECT * FROM non_existent_table")
            
            # Test invalid team name
            result = db.get_team_stats('NonExistentTeam', 2024)
            assert result.empty, "Should return empty result for non-existent team"
            
            db.close()
        except ImportError:
            pytest.skip("Database module not available")
        except Exception as e:
            pytest.fail(f"Database error handling test failed: {e}")
    
    def test_database_statistics(self):
        """Test database statistics function"""
        try:
            from src.storage.simple_database import create_simple_database
            db = create_simple_database()
            
            stats = db.get_database_stats()
            
            # Check required statistics
            required_stats = ['total_records', 'unique_teams', 'seasons', 'conferences']
            for stat in required_stats:
                assert stat in stats, f"Should have {stat} in database stats"
                assert stats[stat] > 0, f"{stat} should be positive"
            
            # Check reasonable values
            assert stats['total_records'] >= 100, "Should have at least 100 records"
            assert stats['unique_teams'] >= 50, "Should have at least 50 teams"
            assert stats['seasons'] >= 1, "Should have at least 1 season"
            assert stats['conferences'] >= 5, "Should have at least 5 conferences"
            
            db.close()
        except ImportError:
            pytest.skip("Database module not available")
        except Exception as e:
            pytest.fail(f"Database statistics test failed: {e}")
    
    @pytest.mark.slow
    def test_database_setup_pipeline(self):
        """Test complete database setup pipeline"""
        from scripts.setup_simple_database import main as setup_database
        
        try:
            # Run database setup
            result = setup_database()
            
            # Verify database was created
            db_path = Path("data/ncaa_football_simple.duckdb")
            assert db_path.exists(), "Database setup should create database file"
            
            # Verify data was loaded
            from src.storage.simple_database import create_simple_database
            db = create_simple_database()
            stats = db.get_database_stats()
            assert stats['total_records'] > 0, "Database should have data after setup"
            db.close()
            
        except Exception as e:
            pytest.fail(f"Database setup pipeline failed: {e}")
    
    def test_database_concurrent_access(self):
        """Test database concurrent access"""
        try:
            from src.storage.simple_database import create_simple_database
            
            # Create multiple connections
            db1 = create_simple_database()
            db2 = create_simple_database()
            
            # Test concurrent queries
            result1 = db1.get_top_teams(2024, 10)
            result2 = db2.get_top_teams(2024, 10)
            
            # Results should be the same
            pd.testing.assert_frame_equal(result1, result2)
            
            # Close connections
            db1.close()
            db2.close()
            
        except ImportError:
            pytest.skip("Database module not available")
        except Exception as e:
            pytest.fail(f"Database concurrent access test failed: {e}")
