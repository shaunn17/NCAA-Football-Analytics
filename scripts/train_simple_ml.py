#!/usr/bin/env python3
"""
Simplified NCAA Football ML Training Script

This script trains basic machine learning models for NCAA football predictions
using the available data (2023-2024 seasons).
"""

import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def prepare_data():
    """Prepare data for ML training"""
    print("📊 Loading and preparing data...")
    
    # Load data
    df = pd.read_csv('data/models/ncaa_football_ml_dataset.csv')
    print(f"   Loaded {len(df)} records")
    
    # Select relevant features
    feature_columns = [
        'win_percentage', 'yards_per_game', 'yards_allowed_per_game',
        'turnover_margin', 'first_down_differential',
        'rushingYards', 'rushingAttempts', 'rushingTDs',
        'netPassingYards', 'passCompletions', 'passAttempts', 'passingTDs',
        'totalYards', 'totalYardsOpponent',
        'sacks', 'tacklesForLoss', 'interceptions', 'fumblesRecovered',
        'thirdDownConversions', 'thirdDowns', 'fourthDownConversions', 'fourthDowns',
        'penalties', 'penaltyYards'
    ]
    
    # Filter to available columns
    available_features = [col for col in feature_columns if col in df.columns]
    features_df = df[available_features + ['team', 'conference', 'year']].copy()
    
    # Handle missing values
    numeric_features = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_features] = features_df[numeric_features].fillna(features_df[numeric_features].median())
    
    # Create derived features
    if all(col in features_df.columns for col in ['totalYards', 'games']):
        features_df['offensive_efficiency'] = features_df['totalYards'] / features_df['games'].replace(0, np.nan)
    
    if all(col in features_df.columns for col in ['totalYardsOpponent', 'games']):
        features_df['defensive_efficiency'] = features_df['totalYardsOpponent'] / features_df['games'].replace(0, np.nan)
    
    if all(col in features_df.columns for col in ['rushingYards', 'rushingAttempts']):
        features_df['rushing_efficiency'] = features_df['rushingYards'] / features_df['rushingAttempts'].replace(0, np.nan)
    
    if all(col in features_df.columns for col in ['netPassingYards', 'passAttempts']):
        features_df['passing_efficiency'] = features_df['netPassingYards'] / features_df['passAttempts'].replace(0, np.nan)
    
    # Fill any new NaN values
    numeric_features = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_features] = features_df[numeric_features].fillna(0)
    
    print(f"   Prepared {len(features_df)} records with {len(available_features)} features")
    return features_df

def create_targets(df):
    """Create target variables"""
    print("🎯 Creating target variables...")
    
    # 1. Top 25 Teams (Binary Classification)
    df['is_top_25'] = 0
    for year in df['year'].unique():
        year_data = df[df['year'] == year]
        if len(year_data) > 25:
            # Top 25 based on win percentage
            top_25_indices = year_data['win_percentage'].nlargest(25).index
            df.loc[top_25_indices, 'is_top_25'] = 1
    
    # 2. Performance Ranking (Regression)
    df['performance_rank'] = 0
    for year in df['year'].unique():
        year_data = df[df['year'] == year]
        if len(year_data) > 0:
            # Rank based on composite score
            composite_score = (
                year_data['win_percentage'] * 0.4 +
                (year_data['yards_per_game'] / year_data['yards_per_game'].max()) * 0.3 +
                (year_data['turnover_margin'] / year_data['turnover_margin'].max()) * 0.3
            )
            ranks = composite_score.rank(ascending=False, method='dense')
            df.loc[df['year'] == year, 'performance_rank'] = ranks
    
    print(f"   Top 25 teams: {df['is_top_25'].sum()}")
    print(f"   Performance rankings created")
    return df

def train_top_25_model(X, y):
    """Train Top 25 prediction model"""
    print("🤖 Training Top 25 prediction model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Try multiple models
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    best_model = None
    best_score = 0
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'model': model
        }
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
    
    print(f"   Best accuracy: {best_score:.3f}")
    return best_model, results

def train_performance_ranking_model(X, y):
    """Train performance ranking model"""
    print("🤖 Training Performance Ranking model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"   R² score: {r2:.3f}, MSE: {mse:.3f}")
    return model, {'mse': mse, 'r2': r2}

def predict_2025_season(df, top_25_model, ranking_model, scaler):
    """Make predictions for 2025 season"""
    print("🔮 Generating 2025 season predictions...")
    
    # Use 2024 data as base for 2025 predictions
    current_data = df[df['year'] == 2024].copy()
    
    if len(current_data) == 0:
        print("   No 2024 data available")
        return pd.DataFrame()
    
    # Prepare features for prediction
    numeric_features = current_data.select_dtypes(include=[np.number]).columns
    ml_features = numeric_features.drop(['year', 'is_top_25', 'performance_rank'], errors='ignore')
    
    X_pred = current_data[ml_features]
    X_pred_scaled = scaler.transform(X_pred)
    
    predictions = current_data[['team', 'conference', 'year']].copy()
    
    # Top 25 predictions
    if top_25_model:
        top_25_proba = top_25_model.predict_proba(X_pred_scaled)[:, 1]
        predictions['top_25_probability'] = top_25_proba
        predictions['predicted_top_25'] = top_25_model.predict(X_pred_scaled)
    
    # Performance ranking predictions
    if ranking_model:
        rank_pred = ranking_model.predict(X_pred_scaled)
        predictions['predicted_rank'] = rank_pred
    
    # Sort by top 25 probability
    if 'top_25_probability' in predictions.columns:
        predictions = predictions.sort_values('top_25_probability', ascending=False)
    
    print(f"   Generated predictions for {len(predictions)} teams")
    return predictions

def main():
    """Main training function"""
    print("🏈 NCAA Football Machine Learning Training")
    print("=" * 50)
    
    try:
        # Prepare data
        df = prepare_data()
        df = create_targets(df)
        
        # Prepare ML features
        numeric_features = df.select_dtypes(include=[np.number]).columns
        ml_features = numeric_features.drop(['year', 'is_top_25', 'performance_rank'], errors='ignore')
        
        X = df[ml_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train models
        top_25_model, top_25_results = train_top_25_model(X_scaled, df['is_top_25'])
        ranking_model, ranking_results = train_performance_ranking_model(X_scaled, df['performance_rank'])
        
        # Generate predictions
        predictions = predict_2025_season(df, top_25_model, ranking_model, scaler)
        
        # Display results
        print("\n🎯 Training Results:")
        print("-" * 30)
        for name, result in top_25_results.items():
            print(f"Top 25 - {name}: {result['accuracy']:.3f} accuracy")
        print(f"Performance Ranking: {ranking_results['r2']:.3f} R² score")
        
        # Show top predictions
        if not predictions.empty:
            print(f"\n🏆 Top 10 Predicted Teams for 2025:")
            print("-" * 40)
            
            top_10 = predictions.head(10)
            for _, row in top_10.iterrows():
                team = row['team']
                conf = row['conference']
                prob = row.get('top_25_probability', 'N/A')
                rank = row.get('predicted_rank', 'N/A')
                
                prob_str = f"{prob:.3f}" if prob != 'N/A' else 'N/A'
                rank_str = f"{rank:.1f}" if rank != 'N/A' else 'N/A'
                print(f"   {team} ({conf}): Top 25 Prob = {prob_str}, Rank = {rank_str}")
        
        # Save models and predictions
        models_dir = Path("data/models")
        models_dir.mkdir(exist_ok=True)
        
        joblib.dump(top_25_model, models_dir / "top_25_model.joblib")
        joblib.dump(ranking_model, models_dir / "ranking_model.joblib")
        joblib.dump(scaler, models_dir / "scaler.joblib")
        
        if not predictions.empty:
            predictions.to_csv(models_dir / "2025_predictions.csv", index=False)
            print(f"\n💾 Models and predictions saved to: {models_dir}")
        
        print("\n✅ ML Training Complete!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
