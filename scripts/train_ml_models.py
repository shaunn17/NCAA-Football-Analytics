#!/usr/bin/env python3
"""
NCAA Football ML Training Script

This script trains machine learning models for NCAA football predictions.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ml.models import NCAAFootballPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main training function"""
    print("🏈 NCAA Football Machine Learning Training")
    print("=" * 50)
    
    try:
        # Initialize predictor
        predictor = NCAAFootballPredictor()
        
        # Train all models
        print("🤖 Training ML models...")
        results = predictor.train_all_models()
        
        # Generate predictions
        print("🔮 Generating 2025 season predictions...")
        predictions = predictor.predict_2025_season()
        
        # Display results
        print("\n🎯 Training Results:")
        print("-" * 30)
        
        for model_name, result in results.items():
            if isinstance(result, dict) and 'test_accuracy' in result:
                print(f"{model_name}: {result['test_accuracy']:.3f} accuracy")
            elif isinstance(result, dict) and 'accuracy' in result:
                print(f"{model_name}: {result['accuracy']:.3f} accuracy")
            elif isinstance(result, dict) and 'r2' in result:
                print(f"{model_name}: {result['r2']:.3f} R² score")
        
        # Show top predictions
        if not predictions.empty:
            print(f"\n🏆 Top 10 Predicted Teams for 2025:")
            print("-" * 40)
            
            display_cols = ['team', 'conference']
            if 'top_25_probability' in predictions.columns:
                display_cols.append('top_25_probability')
            if 'predicted_conference_winner' in predictions.columns:
                display_cols.append('predicted_conference_winner')
            
            top_10 = predictions.head(10)[display_cols]
            for _, row in top_10.iterrows():
                team = row['team']
                conf = row['conference']
                prob = row.get('top_25_probability', 'N/A')
                conf_winner = row.get('predicted_conference_winner', 'N/A')
                
                prob_str = f"{prob:.3f}" if prob != 'N/A' else 'N/A'
                print(f"   {team} ({conf}): Top 25 Prob = {prob_str}, Conf Winner = {conf_winner}")
        
        # Save predictions
        predictions_path = Path("data/models/2025_predictions.csv")
        predictions.to_csv(predictions_path, index=False)
        print(f"\n💾 Predictions saved to: {predictions_path}")
        
        print("\n✅ ML Training Complete!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
