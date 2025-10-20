"""
NCAA Football Machine Learning Models

This module contains machine learning models for predicting:
1. National Champion (Binary Classification)
2. Conference Winners (Multi-class Classification)
3. Team Performance Rankings (Regression)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import logging
from pathlib import Path
from typing import Tuple, Dict, List, Optional

# XGBoost is not available due to OpenMP dependency issues
XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)

class NCAAFootballPredictor:
    """Main class for NCAA Football predictions"""
    
    def __init__(self, data_path: str = "data/models/ncaa_football_ml_dataset.csv"):
        self.data_path = data_path
        self.df = None
        self.features = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.feature_importance = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare the dataset"""
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Loaded dataset with shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def prepare_features(self) -> pd.DataFrame:
        """Prepare features for ML models"""
        if self.df is None:
            self.load_data()
        
        # Select the most complete and relevant features
        feature_columns = [
            # Basic performance metrics
            'win_percentage', 'yards_per_game', 'yards_allowed_per_game',
            'turnover_margin', 'first_down_differential',
            
            # Offensive metrics
            'rushingYards', 'rushingAttempts', 'rushingTDs',
            'netPassingYards', 'passCompletions', 'passAttempts', 'passingTDs',
            'totalYards', 'totalYardsOpponent',
            
            # Defensive metrics
            'sacks', 'tacklesForLoss', 'interceptions', 'fumblesRecovered',
            'sacksOpponent', 'tacklesForLossOpponent',
            
            # Special teams and efficiency
            'thirdDownConversions', 'thirdDowns', 'fourthDownConversions', 'fourthDowns',
            'penalties', 'penaltyYards',
            
            # Conference and team info
            'conference', 'team', 'year'
        ]
        
        # Filter to available columns
        available_features = [col for col in feature_columns if col in self.df.columns]
        self.features = self.df[available_features].copy()
        
        # Handle missing values
        numeric_features = self.features.select_dtypes(include=[np.number]).columns
        self.features[numeric_features] = self.features[numeric_features].fillna(self.features[numeric_features].median())
        
        # Create derived features
        self._create_derived_features()
        
        logger.info(f"Prepared features: {self.features.shape}")
        return self.features
    
    def _create_derived_features(self):
        """Create additional derived features"""
        # Offensive efficiency
        if all(col in self.features.columns for col in ['totalYards', 'games']):
            self.features['offensive_efficiency'] = self.features['totalYards'] / self.features['games'].replace(0, np.nan)
        
        # Defensive efficiency
        if all(col in self.features.columns for col in ['totalYardsOpponent', 'games']):
            self.features['defensive_efficiency'] = self.features['totalYardsOpponent'] / self.features['games'].replace(0, np.nan)
        
        # Rushing efficiency
        if all(col in self.features.columns for col in ['rushingYards', 'rushingAttempts']):
            self.features['rushing_efficiency'] = self.features['rushingYards'] / self.features['rushingAttempts'].replace(0, np.nan)
        
        # Passing efficiency
        if all(col in self.features.columns for col in ['netPassingYards', 'passAttempts']):
            self.features['passing_efficiency'] = self.features['netPassingYards'] / self.features['passAttempts'].replace(0, np.nan)
        
        # Third down conversion rate
        if all(col in self.features.columns for col in ['thirdDownConversions', 'thirdDowns']):
            self.features['third_down_rate'] = self.features['thirdDownConversions'] / self.features['thirdDowns'].replace(0, np.nan)
        
        # Defensive pressure
        if all(col in self.features.columns for col in ['sacks', 'tacklesForLoss']):
            self.features['defensive_pressure'] = self.features['sacks'] + self.features['tacklesForLoss']
        
        # Conference strength (average win percentage by conference)
        if 'conference' in self.features.columns and 'win_percentage' in self.features.columns:
            conference_strength = self.features.groupby('conference')['win_percentage'].transform('mean')
            self.features['conference_strength'] = conference_strength
        
        # Fill any new NaN values
        numeric_features = self.features.select_dtypes(include=[np.number]).columns
        self.features[numeric_features] = self.features[numeric_features].fillna(0)
    
    def create_targets(self) -> pd.DataFrame:
        """Create target variables for different prediction tasks"""
        # Since we only have 2 seasons, let's create targets based on performance rankings
        
        # 1. Top 25 Teams (Binary Classification) - Fixed logic
        self.features['is_top_25'] = 0
        for year in self.features['year'].unique():
            year_data = self.features[self.features['year'] == year]
            if len(year_data) > 25:  # Only if we have more than 25 teams
                # Top 25 based on win percentage
                top_25_indices = year_data['win_percentage'].nlargest(25).index
                self.features.loc[top_25_indices, 'is_top_25'] = 1
        
        # 2. Conference Winner (Multi-class)
        self.features['conference_winner'] = 'None'
        for year in self.features['year'].unique():
            year_data = self.features[self.features['year'] == year]
            for conference in year_data['conference'].unique():
                if pd.notna(conference):
                    conf_data = year_data[year_data['conference'] == conference]
                    if len(conf_data) > 0:
                        winner_idx = conf_data['win_percentage'].idxmax()
                        self.features.loc[winner_idx, 'conference_winner'] = conference
        
        # 3. Performance Ranking (Regression target)
        self.features['performance_rank'] = 0
        for year in self.features['year'].unique():
            year_data = self.features[self.features['year'] == year]
            if len(year_data) > 0:
                # Rank based on composite score
                composite_score = (
                    year_data['win_percentage'] * 0.4 +
                    (year_data['yards_per_game'] / year_data['yards_per_game'].max()) * 0.3 +
                    (year_data['turnover_margin'] / year_data['turnover_margin'].max()) * 0.3
                )
                ranks = composite_score.rank(ascending=False, method='dense')
                self.features.loc[self.features['year'] == year, 'performance_rank'] = ranks
        
        logger.info("Created target variables:")
        logger.info(f"  Top 25 teams: {self.features['is_top_25'].sum()}")
        logger.info(f"  Conference winners: {len(self.features[self.features['conference_winner'] != 'None'])}")
        
        return self.features
    
    def prepare_ml_data(self) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Prepare data for machine learning"""
        self.prepare_features()
        self.create_targets()
        
        # Select numeric features for ML
        numeric_features = self.features.select_dtypes(include=[np.number]).columns
        ml_features = numeric_features.drop(['year', 'is_top_25', 'performance_rank'], errors='ignore')
        
        X = self.features[ml_features]
        
        # Create targets
        targets = {
            'top_25': self.features['is_top_25'],
            'conference_winner': self.features['conference_winner'],
            'performance_rank': self.features['performance_rank']
        }
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        logger.info(f"ML data prepared: {X_scaled.shape}")
        return X_scaled, targets
    
    def train_top_25_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train model to predict top 25 teams"""
        logger.info("Training Top 25 prediction model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Try multiple models
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBClassifier(random_state=42)
        
        best_model = None
        best_score = 0
        results = {}
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            mean_cv_score = cv_scores.mean()
            
            # Train and test
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            
            results[name] = {
                'cv_score': mean_cv_score,
                'test_accuracy': test_accuracy,
                'model': model
            }
            
            if test_accuracy > best_score:
                best_score = test_accuracy
                best_model = model
        
        # Store best model
        self.models['top_25'] = best_model
        self.feature_importance['top_25'] = best_model.feature_importances_ if hasattr(best_model, 'feature_importances_') else None
        
        logger.info(f"Best Top 25 model accuracy: {best_score:.3f}")
        return results
    
    def train_conference_winner_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train model to predict conference winners"""
        logger.info("Training Conference Winner prediction model...")
        
        # Filter out 'None' labels for training
        valid_mask = y != 'None'
        X_filtered = X[valid_mask]
        y_filtered = y[valid_mask]
        
        if len(y_filtered) == 0:
            logger.warning("No valid conference winner data for training")
            return {}
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_filtered)
        self.label_encoders['conference_winner'] = le
        
        # Split data without stratification for limited data
        if len(y_encoded) > 20:  # Only stratify if we have enough samples
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_encoded, 
                test_size=0.2, 
                random_state=42, 
                stratify=y_encoded
            )
        else:
            # Use simple split for small datasets
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_encoded, 
                test_size=0.2, 
                random_state=42
            )
        
        # Train Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.models['conference_winner'] = model
        self.feature_importance['conference_winner'] = model.feature_importances_
        
        logger.info(f"Conference Winner model accuracy: {accuracy:.3f}")
        return {'accuracy': accuracy, 'model': model}
    
    def train_performance_ranking_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train model to predict performance ranking"""
        logger.info("Training Performance Ranking model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.models['performance_rank'] = model
        self.feature_importance['performance_rank'] = model.feature_importances_
        
        logger.info(f"Performance Ranking model R²: {r2:.3f}, MSE: {mse:.3f}")
        return {'mse': mse, 'r2': r2, 'model': model}
    
    def train_all_models(self) -> Dict:
        """Train all ML models"""
        logger.info("Starting ML model training...")
        
        X, targets = self.prepare_ml_data()
        
        results = {}
        
        # Train Top 25 model
        if 'top_25' in targets:
            results['top_25'] = self.train_top_25_model(X, targets['top_25'])
        
        # Train Conference Winner model
        if 'conference_winner' in targets:
            results['conference_winner'] = self.train_conference_winner_model(X, targets['conference_winner'])
        
        # Train Performance Ranking model
        if 'performance_rank' in targets:
            results['performance_rank'] = self.train_performance_ranking_model(X, targets['performance_rank'])
        
        # Save models
        self.save_models()
        
        logger.info("All models trained successfully!")
        return results
    
    def predict_2025_season(self) -> pd.DataFrame:
        """Make predictions for 2025 season"""
        logger.info("Making predictions for 2025 season...")
        
        if not self.models:
            logger.error("No trained models available. Please train models first.")
            return pd.DataFrame()
        
        # Use 2024 data as base for 2025 predictions
        current_data = self.features[self.features['year'] == 2024].copy()
        
        if len(current_data) == 0:
            logger.error("No 2024 data available for predictions")
            return pd.DataFrame()
        
        # Prepare features for prediction
        numeric_features = current_data.select_dtypes(include=[np.number]).columns
        ml_features = numeric_features.drop(['year', 'is_top_25', 'performance_rank'], errors='ignore')
        
        X_pred = current_data[ml_features]
        X_pred_scaled = pd.DataFrame(
            self.scaler.transform(X_pred),
            columns=X_pred.columns,
            index=X_pred.index
        )
        
        predictions = current_data[['team', 'conference', 'year']].copy()
        
        # Top 25 predictions
        if 'top_25' in self.models:
            top_25_proba = self.models['top_25'].predict_proba(X_pred_scaled)[:, 1]
            predictions['top_25_probability'] = top_25_proba
            predictions['predicted_top_25'] = self.models['top_25'].predict(X_pred_scaled)
        
        # Conference winner predictions
        if 'conference_winner' in self.models:
            conf_winner_pred = self.models['conference_winner'].predict(X_pred_scaled)
            predictions['predicted_conference_winner'] = self.label_encoders['conference_winner'].inverse_transform(conf_winner_pred)
        
        # Performance ranking predictions
        if 'performance_rank' in self.models:
            rank_pred = self.models['performance_rank'].predict(X_pred_scaled)
            predictions['predicted_rank'] = rank_pred
        
        # Sort by top 25 probability
        if 'top_25_probability' in predictions.columns:
            predictions = predictions.sort_values('top_25_probability', ascending=False)
        
        logger.info(f"Generated predictions for {len(predictions)} teams")
        return predictions
    
    def save_models(self):
        """Save trained models"""
        models_dir = Path("data/models")
        models_dir.mkdir(exist_ok=True)
        
        for name, model in self.models.items():
            model_path = models_dir / f"{name}_model.joblib"
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} model to {model_path}")
        
        # Save scaler and encoders
        joblib.dump(self.scaler, models_dir / "scaler.joblib")
        joblib.dump(self.label_encoders, models_dir / "label_encoders.joblib")
    
    def load_models(self):
        """Load trained models"""
        models_dir = Path("data/models")
        
        model_files = {
            'top_25': 'top_25_model.joblib',
            'conference_winner': 'conference_winner_model.joblib',
            'performance_rank': 'performance_rank_model.joblib'
        }
        
        for name, filename in model_files.items():
            model_path = models_dir / filename
            if model_path.exists():
                self.models[name] = joblib.load(model_path)
                logger.info(f"Loaded {name} model")
        
        # Load scaler and encoders
        scaler_path = models_dir / "scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        
        encoders_path = models_dir / "label_encoders.joblib"
        if encoders_path.exists():
            self.label_encoders = joblib.load(encoders_path)

if __name__ == "__main__":
    # Example usage
    predictor = NCAAFootballPredictor()
    results = predictor.train_all_models()
    predictions = predictor.predict_2025_season()
    
    print("🏈 NCAA Football ML Models Training Complete!")
    print(f"📊 Predictions generated for {len(predictions)} teams")
    print("\n🎯 Top 10 Predicted Teams for 2025:")
    print(predictions.head(10)[['team', 'conference', 'top_25_probability']].to_string(index=False))
