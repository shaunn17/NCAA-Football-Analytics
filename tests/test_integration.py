"""
Integration Tests for NCAA Football Analytics

Tests to ensure all components work together correctly.
"""

import pytest
import pandas as pd
from pathlib import Path
from tests.conftest import assert_dataframe_not_empty, assert_column_exists

class TestIntegration:
    """Test integration between different components"""
    
    def test_data_pipeline_integration(self):
        """Test complete data pipeline integration"""
        # Check that all pipeline outputs exist
        required_files = [
            "data/models/ncaa_football_ml_dataset.csv",
            "data/models/2025_predictions.csv"
        ]
        
        for file_path in required_files:
            path = Path(file_path)
            if path.exists():
                df = pd.read_csv(path)
                assert_dataframe_not_empty(df, f"Pipeline output: {file_path}")
            else:
                pytest.skip(f"Pipeline output not available: {file_path}")
    
    def test_dashboard_data_integration(self):
        """Test dashboard integration with data"""
        try:
            # Test that dashboard can load data
            from simple_dashboard import load_data, load_predictions
            
            # Load data
            df = load_data()
            assert_dataframe_not_empty(df, "Dashboard data")
            
            # Check required columns for dashboard
            required_columns = ['team', 'year', 'conference', 'win_percentage']
            for col in required_columns:
                assert_column_exists(df, col, "Dashboard data")
            
            # Load predictions
            predictions = load_predictions()
            if not predictions.empty:
                assert_column_exists(predictions, 'team', "Dashboard predictions")
                assert_column_exists(predictions, 'top_25_probability', "Dashboard predictions")
            
        except ImportError:
            pytest.skip("Dashboard module not available")
        except Exception as e:
            pytest.fail(f"Dashboard data integration failed: {e}")
    
    def test_database_dashboard_integration(self):
        """Test database integration with dashboard"""
        try:
            from src.storage.simple_database import create_simple_database
            
            db = create_simple_database()
            
            # Test dashboard query functions
            top_teams = db.get_top_teams(2024, 10)
            assert_dataframe_not_empty(top_teams, "Dashboard top teams query")
            
            # Test team comparison
            comparison = db.get_team_comparison('Indiana', 'Ohio State', 2024)
            if not comparison.empty:
                assert_column_exists(comparison, 'team', "Team comparison")
                assert_column_exists(comparison, 'win_percentage', "Team comparison")
            
            db.close()
            
        except ImportError:
            pytest.skip("Database module not available")
        except Exception as e:
            pytest.fail(f"Database dashboard integration failed: {e}")
    
    def test_ml_pipeline_integration(self):
        """Test ML pipeline integration"""
        # Check ML pipeline outputs
        ml_files = [
            "data/models/2025_predictions.csv"
        ]
        
        for file_path in ml_files:
            path = Path(file_path)
            if path.exists():
                df = pd.read_csv(path)
                assert_dataframe_not_empty(df, f"ML output: {file_path}")
                
                # Check ML-specific columns
                if 'top_25_probability' in df.columns:
                    probabilities = df['top_25_probability'].dropna()
                    if len(probabilities) > 0:
                        assert all(0 <= p <= 1 for p in probabilities), "ML probabilities should be between 0 and 1"
            else:
                pytest.skip(f"ML output not available: {file_path}")
    
    def test_api_data_integration(self):
        """Test API integration with data processing"""
        try:
            from src.ingestion.api_client import CollegeFootballAPIClient
            from src.ingestion.data_collector import DataCollector
            
            # Test API client
            client = CollegeFootballAPIClient()
            assert client.api_key is not None, "API client should have key"
            
            # Test data collector
            collector = DataCollector()
            assert collector.api_client is not None, "Data collector should have API client"
            
        except ImportError:
            pytest.skip("API modules not available")
        except Exception as e:
            pytest.fail(f"API data integration failed: {e}")
    
    def test_data_processing_integration(self):
        """Test data processing pipeline integration"""
        try:
            from src.processing.cleaner import DataCleaner
            from src.processing.transformer import DataTransformer
            
            # Test cleaner
            cleaner = DataCleaner()
            assert cleaner.raw_data_dir.exists(), "Cleaner should have data directory"
            
            # Test transformer
            transformer = DataTransformer()
            assert transformer.processed_data_dir.exists(), "Transformer should have data directory"
            
        except ImportError:
            pytest.skip("Processing modules not available")
        except Exception as e:
            pytest.fail(f"Data processing integration failed: {e}")
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # Check that all major components are working together
        
        # 1. Data exists
        data_path = Path("data/models/ncaa_football_ml_dataset.csv")
        if not data_path.exists():
            pytest.skip("Processed data not available")
        
        df = pd.read_csv(data_path)
        assert_dataframe_not_empty(df, "End-to-end data")
        
        # 2. Database works with data
        try:
            from src.storage.simple_database import create_simple_database
            db = create_simple_database()
            stats = db.get_database_stats()
            assert stats['total_records'] > 0, "Database should have data"
            db.close()
        except ImportError:
            pass  # Database optional
        
        # 3. ML predictions exist
        predictions_path = Path("data/models/2025_predictions.csv")
        if predictions_path.exists():
            predictions = pd.read_csv(predictions_path)
            assert_dataframe_not_empty(predictions, "End-to-end predictions")
        
        # 4. Dashboard can load everything
        try:
            from simple_dashboard import load_data, load_predictions
            
            dashboard_data = load_data()
            assert_dataframe_not_empty(dashboard_data, "End-to-end dashboard data")
            
            dashboard_predictions = load_predictions()
            # Predictions might be empty, that's okay
            
        except ImportError:
            pass  # Dashboard optional
    
    def test_data_consistency_across_components(self):
        """Test data consistency across all components"""
        # Load data from different sources
        data_sources = []
        
        # From CSV
        csv_path = Path("data/models/ncaa_football_ml_dataset.csv")
        if csv_path.exists():
            csv_data = pd.read_csv(csv_path)
            data_sources.append(("CSV", csv_data))
        
        # From database
        try:
            from src.storage.simple_database import create_simple_database
            db = create_simple_database()
            db_data = db.query("SELECT * FROM ncaa_football_data")
            data_sources.append(("Database", db_data))
            db.close()
        except ImportError:
            pass
        
        # Compare data consistency
        if len(data_sources) >= 2:
            source1_name, source1_data = data_sources[0]
            source2_name, source2_data = data_sources[1]
            
            # Check that both have same teams
            teams1 = set(source1_data['team'].unique())
            teams2 = set(source2_data['team'].unique())
            
            # Should have significant overlap
            overlap = len(teams1.intersection(teams2))
            total_teams = len(teams1.union(teams2))
            
            overlap_ratio = overlap / total_teams if total_teams > 0 else 0
            assert overlap_ratio > 0.8, f"Data sources should have >80% team overlap, got {overlap_ratio:.2f}"
    
    def test_error_handling_integration(self):
        """Test error handling across components"""
        # Test that components handle missing data gracefully
        
        # Test dashboard with missing data
        try:
            from simple_dashboard import load_data
            
            # Should not crash even if data is missing
            data = load_data()
            # Empty data is okay, just shouldn't crash
            
        except ImportError:
            pass
        except Exception as e:
            pytest.fail(f"Dashboard should handle missing data gracefully: {e}")
        
        # Test database with missing data
        try:
            from src.storage.simple_database import create_simple_database
            db = create_simple_database()
            
            # Should handle invalid queries gracefully
            result = db.get_team_stats('NonExistentTeam', 2024)
            assert result.empty, "Should return empty result for non-existent team"
            
            db.close()
            
        except ImportError:
            pass
        except Exception as e:
            pytest.fail(f"Database should handle missing data gracefully: {e}")
    
    @pytest.mark.slow
    def test_performance_integration(self):
        """Test performance across integrated components"""
        import time
        
        # Test dashboard loading performance
        try:
            from simple_dashboard import load_data
            
            start_time = time.time()
            data = load_data()
            load_time = time.time() - start_time
            
            assert load_time < 5, f"Dashboard data loading took {load_time:.2f}s, should be < 5s"
            
        except ImportError:
            pass
        
        # Test database query performance
        try:
            from src.storage.simple_database import create_simple_database
            db = create_simple_database()
            
            start_time = time.time()
            top_teams = db.get_top_teams(2024, 25)
            query_time = time.time() - start_time
            
            assert query_time < 2, f"Database query took {query_time:.2f}s, should be < 2s"
            
            db.close()
            
        except ImportError:
            pass
