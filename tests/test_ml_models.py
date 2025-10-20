"""
ML Model Tests for NCAA Football Analytics

Tests to ensure machine learning models work correctly and meet performance standards.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from tests.conftest import assert_dataframe_not_empty, assert_column_exists, assert_numeric_range

class TestMLModels:
    """Test machine learning models and predictions"""
    
    def test_ml_dataset_exists(self):
        """Test that ML dataset exists and is valid"""
        data_path = Path("data/models/ncaa_football_ml_dataset.csv")
        assert data_path.exists(), "ML dataset should exist"
        
        df = pd.read_csv(data_path)
        assert_dataframe_not_empty(df, "ML dataset")
        
        # Check for required ML features
        required_features = ['team', 'year', 'win_percentage', 'yards_per_game']
        for feature in required_features:
            assert_column_exists(df, feature, "ML dataset")
    
    def test_ml_predictions_exist(self):
        """Test that ML predictions exist and are valid"""
        predictions_path = Path("data/models/2025_predictions.csv")
        if not predictions_path.exists():
            pytest.skip("ML predictions not available")
        
        df = pd.read_csv(predictions_path)
        assert_dataframe_not_empty(df, "ML predictions")
        
        # Check required prediction columns
        required_columns = ['team', 'top_25_probability']
        for col in required_columns:
            assert_column_exists(df, col, "ML predictions")
    
    def test_top_25_probability_range(self):
        """Test that top 25 probabilities are in valid range"""
        predictions_path = Path("data/models/2025_predictions.csv")
        if not predictions_path.exists():
            pytest.skip("ML predictions not available")
        
        df = pd.read_csv(predictions_path)
        
        if 'top_25_probability' in df.columns:
            probabilities = df['top_25_probability'].dropna()
            if len(probabilities) > 0:
                assert_numeric_range(probabilities, 0, 1, "Top 25 probabilities")
    
    def test_ml_model_training(self, sample_team_data):
        """Test ML model training process"""
        from src.ml.models import NCAAFootballPredictor
        
        # Create predictor with sample data
        predictor = NCAAFootballPredictor()
        predictor.features = sample_team_data
        
        # Test target creation
        targets_df = predictor.create_targets()
        assert_dataframe_not_empty(targets_df, "Target variables")
        
        # Check that targets were created
        assert 'is_top_25' in targets_df.columns, "Should create top 25 target"
        assert 'performance_rank' in targets_df.columns, "Should create performance rank target"
    
    def test_top_25_model_accuracy(self, sample_team_data):
        """Test top 25 prediction model accuracy"""
        from src.ml.models import NCAAFootballPredictor
        
        predictor = NCAAFootballPredictor()
        predictor.features = sample_team_data
        
        # Create targets
        targets_df = predictor.create_targets()
        
        # Prepare features for training
        feature_cols = ['win_percentage', 'yards_per_game', 'yards_allowed_per_game', 'turnover_margin']
        available_cols = [col for col in feature_cols if col in targets_df.columns]
        
        if len(available_cols) < 2:
            pytest.skip("Not enough features for ML testing")
        
        X = targets_df[available_cols].fillna(0)
        y = targets_df['is_top_25']
        
        # Train model
        try:
            results = predictor.train_top_25_model(X, y)
            
            # Check that model was trained successfully
            assert 'model' in results, "Should return trained model"
            assert 'accuracy' in results, "Should return accuracy score"
            
            # Check accuracy is reasonable (at least better than random)
            accuracy = results['accuracy']
            assert accuracy > 0.5, f"Model accuracy {accuracy:.3f} should be better than random (0.5)"
            
        except Exception as e:
            pytest.fail(f"Top 25 model training failed: {e}")
    
    def test_performance_ranking_model(self, sample_team_data):
        """Test performance ranking model"""
        from src.ml.models import NCAAFootballPredictor
        
        predictor = NCAAFootballPredictor()
        predictor.features = sample_team_data
        
        # Create targets
        targets_df = predictor.create_targets()
        
        # Prepare features for training
        feature_cols = ['win_percentage', 'yards_per_game', 'yards_allowed_per_game', 'turnover_margin']
        available_cols = [col for col in feature_cols if col in targets_df.columns]
        
        if len(available_cols) < 2:
            pytest.skip("Not enough features for ML testing")
        
        X = targets_df[available_cols].fillna(0)
        y = targets_df['performance_rank']
        
        # Train model
        try:
            results = predictor.train_performance_ranking_model(X, y)
            
            # Check that model was trained successfully
            assert 'model' in results, "Should return trained model"
            assert 'r2_score' in results, "Should return R² score"
            
            # Check R² score is reasonable
            r2 = results['r2_score']
            assert r2 > 0.5, f"Model R² score {r2:.3f} should be > 0.5"
            
        except Exception as e:
            pytest.fail(f"Performance ranking model training failed: {e}")
    
    def test_model_predictions_consistency(self):
        """Test that model predictions are consistent"""
        predictions_path = Path("data/models/2025_predictions.csv")
        if not predictions_path.exists():
            pytest.skip("ML predictions not available")
        
        df = pd.read_csv(predictions_path)
        
        # Test that predictions are consistent with team performance
        if 'top_25_probability' in df.columns and 'performance_rank' in df.columns:
            # Teams with higher performance rank should generally have higher top 25 probability
            correlation = df['top_25_probability'].corr(df['performance_rank'])
            
            # Should be negative correlation (higher rank = lower number = higher probability)
            assert correlation < 0, f"Performance rank and top 25 probability should be negatively correlated, got {correlation:.3f}"
    
    def test_model_feature_importance(self, sample_team_data):
        """Test that model uses meaningful features"""
        from src.ml.models import NCAAFootballPredictor
        
        predictor = NCAAFootballPredictor()
        predictor.features = sample_team_data
        
        # Create targets
        targets_df = predictor.create_targets()
        
        # Prepare features
        feature_cols = ['win_percentage', 'yards_per_game', 'yards_allowed_per_game', 'turnover_margin']
        available_cols = [col for col in feature_cols if col in targets_df.columns]
        
        if len(available_cols) < 2:
            pytest.skip("Not enough features for ML testing")
        
        X = targets_df[available_cols].fillna(0)
        y = targets_df['is_top_25']
        
        # Train model and check feature importance
        try:
            results = predictor.train_top_25_model(X, y)
            
            if 'feature_importance' in results:
                importance = results['feature_importance']
                
                # Check that win_percentage has high importance
                if 'win_percentage' in importance:
                    win_importance = importance['win_percentage']
                    assert win_importance > 0.1, f"Win percentage should have meaningful importance, got {win_importance:.3f}"
            
        except Exception as e:
            pytest.fail(f"Feature importance testing failed: {e}")
    
    def test_model_prediction_distribution(self):
        """Test that model predictions have reasonable distribution"""
        predictions_path = Path("data/models/2025_predictions.csv")
        if not predictions_path.exists():
            pytest.skip("ML predictions not available")
        
        df = pd.read_csv(predictions_path)
        
        if 'top_25_probability' in df.columns:
            probabilities = df['top_25_probability'].dropna()
            
            if len(probabilities) > 0:
                # Check distribution properties
                mean_prob = probabilities.mean()
                std_prob = probabilities.std()
                
                # Mean should be reasonable (not too high or too low)
                assert 0.1 < mean_prob < 0.9, f"Mean probability {mean_prob:.3f} should be between 0.1 and 0.9"
                
                # Should have some variation
                assert std_prob > 0.05, f"Standard deviation {std_prob:.3f} should show variation"
    
    def test_model_cross_validation(self, sample_team_data):
        """Test model performance with cross-validation"""
        from src.ml.models import NCAAFootballPredictor
        from sklearn.model_selection import cross_val_score
        
        predictor = NCAAFootballPredictor()
        predictor.features = sample_team_data
        
        # Create targets
        targets_df = predictor.create_targets()
        
        # Prepare features
        feature_cols = ['win_percentage', 'yards_per_game', 'yards_allowed_per_game', 'turnover_margin']
        available_cols = [col for col in feature_cols if col in targets_df.columns]
        
        if len(available_cols) < 2:
            pytest.skip("Not enough features for ML testing")
        
        X = targets_df[available_cols].fillna(0)
        y = targets_df['is_top_25']
        
        # Only test if we have enough samples for cross-validation
        if len(X) < 10:
            pytest.skip("Not enough samples for cross-validation")
        
        try:
            # Train model
            results = predictor.train_top_25_model(X, y)
            model = results['model']
            
            # Test cross-validation
            cv_scores = cross_val_score(model, X, y, cv=min(3, len(X)//2))
            
            # Check that cross-validation scores are reasonable
            mean_cv_score = cv_scores.mean()
            assert mean_cv_score > 0.4, f"Cross-validation accuracy {mean_cv_score:.3f} should be > 0.4"
            
            # Check that scores are consistent (low variance)
            cv_std = cv_scores.std()
            assert cv_std < 0.3, f"Cross-validation variance {cv_std:.3f} should be < 0.3"
            
        except Exception as e:
            pytest.fail(f"Cross-validation testing failed: {e}")
    
    @pytest.mark.slow
    def test_full_ml_pipeline(self):
        """Test complete ML pipeline from data to predictions"""
        from scripts.train_ml_models import main as train_models
        
        try:
            # Run ML training pipeline
            result = train_models()
            
            # Verify outputs exist
            predictions_path = Path("data/models/2025_predictions.csv")
            assert predictions_path.exists(), "ML pipeline should create predictions file"
            
            # Verify predictions are valid
            df = pd.read_csv(predictions_path)
            assert_dataframe_not_empty(df, "ML predictions")
            
            # Check required columns
            assert_column_exists(df, 'team', "ML predictions")
            assert_column_exists(df, 'top_25_probability', "ML predictions")
            
        except Exception as e:
            pytest.fail(f"Full ML pipeline failed: {e}")
    
    def test_model_serialization(self, sample_team_data):
        """Test that models can be saved and loaded"""
        from src.ml.models import NCAAFootballPredictor
        import tempfile
        
        predictor = NCAAFootballPredictor()
        predictor.features = sample_team_data
        
        # Create targets
        targets_df = predictor.create_targets()
        
        # Prepare features
        feature_cols = ['win_percentage', 'yards_per_game', 'yards_allowed_per_game', 'turnover_margin']
        available_cols = [col for col in feature_cols if col in targets_df.columns]
        
        if len(available_cols) < 2:
            pytest.skip("Not enough features for ML testing")
        
        X = targets_df[available_cols].fillna(0)
        y = targets_df['is_top_25']
        
        try:
            # Train model
            results = predictor.train_top_25_model(X, y)
            model = results['model']
            
            # Test serialization
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                joblib.dump(model, f.name)
                
                # Load model back
                loaded_model = joblib.load(f.name)
                
                # Test that loaded model works
                predictions_original = model.predict(X)
                predictions_loaded = loaded_model.predict(X)
                
                # Should give same predictions
                np.testing.assert_array_equal(predictions_original, predictions_loaded)
                
                # Clean up
                Path(f.name).unlink()
                
        except Exception as e:
            pytest.fail(f"Model serialization testing failed: {e}")
