# 📖 API Documentation

This guide provides comprehensive documentation for programmatic access to the NCAA Football Analytics Platform.

## 🚀 Overview

The platform provides several APIs for different use cases:

1. **Data Processing API**: Access to data collection and processing
2. **Analytics API**: Advanced analytics and statistical functions
3. **ML API**: Machine learning model training and predictions
4. **Database API**: Direct database access and queries

## 📊 Data Processing API

### **Data Collection**

#### **CollegeFootballAPIClient**
Main client for interacting with the College Football Data API.

```python
from src.ingestion.api_client import CollegeFootballAPIClient

# Initialize client
client = CollegeFootballAPIClient(api_key="your_api_key")

# Get all teams
teams = client.get_teams()
print(teams.head())

# Get team statistics for a year
team_stats = client.get_team_stats(year=2024)
print(team_stats.head())

# Get games data
games = client.get_games(year=2024)
print(games.head())
```

#### **DataCollector**
Orchestrates data collection from multiple sources.

```python
from src.ingestion.data_collector import DataCollector

# Initialize collector
collector = DataCollector(api_key="your_api_key")

# Collect all data
collector.collect_all_data(
    start_year=2023,
    end_year=2024,
    conferences=["Big Ten", "SEC", "ACC"]
)
```

### **Data Processing**

#### **DataCleaner**
Cleans and standardizes raw data.

```python
from src.processing.cleaner import DataCleaner

# Initialize cleaner
cleaner = DataCleaner()

# Run cleaning pipeline
cleaned_data = cleaner.run_cleaning_pipeline(
    start_year=2023,
    end_year=2024
)

print(cleaned_data.head())
```

#### **DataTransformer**
Transforms cleaned data into ML-ready features.

```python
from src.processing.transformer import DataTransformer

# Initialize transformer
transformer = DataTransformer(cleaned_data)

# Run transformation pipeline
ml_dataset = transformer.run_transformation_pipeline()

print(ml_dataset.head())
```

## 🔬 Analytics API

### **Advanced Analytics**

#### **AdvancedAnalytics**
Main analytics engine for statistical analysis.

```python
from src.analytics import AdvancedAnalytics
import pandas as pd

# Load data
data = pd.read_csv("data/models/ncaa_football_ml_dataset.csv")

# Initialize analytics engine
analytics = AdvancedAnalytics(data)

# Trend analysis
trend_data = analytics.calculate_trend_analysis(
    team="Ohio State",
    metric="win_percentage",
    years=[2023, 2024]
)

print(f"Trend Direction: {trend_data['trend_direction']}")
print(f"R² Score: {trend_data['r_squared']:.3f}")
```

#### **Performance Clustering**
Group teams by performance characteristics.

```python
# Performance clustering
cluster_data = analytics.calculate_performance_clusters(
    year=2024,
    metrics=['win_percentage', 'yards_per_game', 'turnover_margin']
)

print("Clusters:")
for cluster_name, cluster_info in cluster_data['clusters'].items():
    print(f"{cluster_name}: {cluster_info['count']} teams")
```

#### **Statistical Comparison**
Compare teams using statistical significance tests.

```python
# Statistical comparison
comparison_data = analytics.calculate_statistical_significance(
    team1="Ohio State",
    team2="Michigan",
    metric="win_percentage",
    year=2024
)

print(f"P-Value: {comparison_data['p_value']:.4f}")
print(f"Significance: {comparison_data['significance']}")
```

#### **Predictive Insights**
Generate predictions for future performance.

```python
# Predictive insights
predictions_data = analytics.calculate_predictive_insights(
    team="Ohio State",
    year=2025
)

print(f"Overall Trajectory: {predictions_data['overall_trajectory']}")
print(f"Confidence Level: {predictions_data['confidence_level']}")
```

#### **Conference Analysis**
Analyze conference-wide performance.

```python
# Conference analysis
conference_data = analytics.calculate_conference_analysis(
    conference="Big Ten",
    year=2024
)

print(f"Competitive Balance: {conference_data['competitive_balance']}")
print(f"Conference Style: {conference_data['conference_style']}")
```

#### **Team Insights Summary**
Generate comprehensive team analysis.

```python
# Team insights summary
insights_data = analytics.generate_insights_summary(
    team="Ohio State",
    year=2024
)

print("Strengths:")
for strength in insights_data['strengths']:
    print(f"  - {strength}")

print("Recommendations:")
for recommendation in insights_data['recommendations']:
    print(f"  - {recommendation}")
```

### **Advanced Visualizations**

#### **AdvancedVisualizations**
Create sophisticated visualizations for analytics results.

```python
from src.analytics import AdvancedVisualizations

# Trend analysis chart
fig = AdvancedVisualizations.create_trend_analysis_chart(trend_data)
fig.show()

# Performance clustering chart
fig = AdvancedVisualizations.create_performance_cluster_chart(cluster_data)
fig.show()

# Statistical comparison chart
fig = AdvancedVisualizations.create_statistical_comparison_chart(comparison_data)
fig.show()
```

## 🤖 Machine Learning API

### **Model Training**

#### **NCAAFootballPredictor**
Main ML model training and prediction engine.

```python
from src.ml.models import NCAAFootballPredictor
import pandas as pd

# Load data
data = pd.read_csv("data/models/ncaa_football_ml_dataset.csv")

# Initialize predictor
predictor = NCAAFootballPredictor(data)

# Create targets
features_with_targets = predictor.create_targets()

# Train Top 25 model
X_top_25 = features_with_targets.drop(columns=['is_top_25', 'conference_winner', 'performance_rank', 'team', 'conference'], errors='ignore')
y_top_25 = features_with_targets['is_top_25']

top_25_results = predictor.train_top_25_model(X_top_25, y_top_25)
print(f"Top 25 Model Accuracy: {top_25_results['accuracy']:.3f}")

# Train Performance Ranking model
X_perf = features_with_targets.drop(columns=['is_top_25', 'conference_winner', 'performance_rank', 'team', 'conference'], errors='ignore')
y_perf = features_with_targets['performance_rank']

perf_results = predictor.train_performance_ranking_model(X_perf, y_perf)
print(f"Performance Model R²: {perf_results['r2_score']:.3f}")
```

### **Model Predictions**

#### **Generate Predictions**
Create predictions for new data.

```python
# Generate predictions for all teams
predictions = predictor.predict_all_teams()

print("Top 10 Teams by Top 25 Probability:")
top_teams = predictions.nlargest(10, 'top_25_probability')
print(top_teams[['team', 'top_25_probability', 'performance_rank']])
```

#### **Save and Load Models**
Persist trained models for later use.

```python
# Save models
predictor.save_models("data/models/")

# Load models
predictor.load_models("data/models/")

# Use loaded models for predictions
predictions = predictor.predict_all_teams()
```

## 🗄️ Database API

### **Database Connection**

#### **SimpleDatabaseManager**
Direct access to the DuckDB database.

```python
from src.storage.simple_database import SimpleDatabaseManager
from pathlib import Path

# Initialize database manager
db = SimpleDatabaseManager(Path("data"))

# Get team statistics
team_stats = db.get_team_stats("Ohio State", 2024)
print(team_stats)

# Get conference standings
standings = db.get_conference_standings("Big Ten", 2024)
print(standings)

# Get top teams
top_teams = db.get_top_teams(2024, 10)
print(top_teams)

# Get team comparison
comparison = db.get_team_comparison("Ohio State", "Michigan", 2024)
print(comparison)
```

### **Custom Queries**

#### **Direct SQL Queries**
Execute custom SQL queries on the database.

```python
# Custom query
query = """
SELECT team, year, win_percentage, yards_per_game
FROM ncaa_football_data
WHERE conference = 'Big Ten'
AND year = 2024
ORDER BY win_percentage DESC
"""

results = db.query(query)
print(results)
```

#### **Database Statistics**
Get database performance and statistics.

```python
# Database statistics
stats = db.get_database_stats()
print(f"Total Records: {stats['total_records']}")
print(f"Unique Teams: {stats['unique_teams']}")
print(f"Seasons: {stats['seasons']}")
print(f"Conferences: {stats['conferences']}")
```

## 🏭 Production Pipeline API

### **Pipeline Management**

#### **ProductionPipeline**
Manage the production data pipeline.

```python
from src.pipeline import ProductionPipeline, PipelineConfig

# Load configuration
config = PipelineConfig(
    start_year=2023,
    end_year=2024,
    schedule_interval="daily",
    schedule_time="06:00"
)

# Initialize pipeline
pipeline = ProductionPipeline(config)

# Run pipeline once
success = pipeline.run_pipeline()
print(f"Pipeline Success: {success}")

# Start scheduler
pipeline.start_scheduler()
```

### **Pipeline Monitoring**

#### **Pipeline Status**
Monitor pipeline status and performance.

```python
# Check pipeline status
print(f"Status: {pipeline.status.status}")
print(f"Duration: {pipeline.status.duration}")
print(f"Records Processed: {pipeline.status.records_processed}")
print(f"Data Quality Score: {pipeline.status.data_quality_score}")

# Check for errors
if pipeline.status.errors:
    print("Errors:")
    for error in pipeline.status.errors:
        print(f"  - {error}")

# Check for warnings
if pipeline.status.warnings:
    print("Warnings:")
    for warning in pipeline.status.warnings:
        print(f"  - {warning}")
```

## 📊 Data Export API

### **Export Data**

#### **Export to CSV**
Export data to CSV format.

```python
import pandas as pd

# Load data
data = pd.read_csv("data/models/ncaa_football_ml_dataset.csv")

# Export specific data
big_ten_teams = data[data['conference'] == 'Big Ten']
big_ten_teams.to_csv("big_ten_teams.csv", index=False)

# Export predictions
predictions = pd.read_csv("data/models/2025_predictions.csv")
top_25_predictions = predictions.nlargest(25, 'top_25_probability')
top_25_predictions.to_csv("top_25_predictions.csv", index=False)
```

#### **Export to JSON**
Export data to JSON format.

```python
import json

# Export team data
team_data = data[data['team'] == 'Ohio State'].to_dict('records')
with open('ohio_state_data.json', 'w') as f:
    json.dump(team_data, f, indent=2)
```

#### **Export to Excel**
Export data to Excel format.

```python
# Export multiple sheets
with pd.ExcelWriter('ncaa_football_data.xlsx') as writer:
    data.to_excel(writer, sheet_name='All Teams', index=False)
    predictions.to_excel(writer, sheet_name='Predictions', index=False)
    standings.to_excel(writer, sheet_name='Standings', index=False)
```

## 🔧 Configuration API

### **Settings Management**

#### **Settings Configuration**
Access and modify platform settings.

```python
from config.settings import settings

# Access settings
print(f"API Key: {settings.cfbd_api_key}")
print(f"Raw Data Dir: {settings.raw_data_dir}")
print(f"Processed Data Dir: {settings.processed_data_dir}")

# Modify settings (for development)
settings.raw_data_dir = Path("custom/raw")
settings.processed_data_dir = Path("custom/processed")
```

### **Pipeline Configuration**

#### **Pipeline Configuration**
Configure the production pipeline.

```python
from src.pipeline import PipelineConfig

# Create custom configuration
config = PipelineConfig(
    start_year=2023,
    end_year=2024,
    conferences=["Big Ten", "SEC", "ACC"],
    schedule_interval="weekly",
    schedule_time="06:00",
    enable_notifications=True,
    notification_email="admin@example.com",
    min_data_quality_score=0.8,
    enable_backup=True,
    backup_retention_days=30
)

# Save configuration
import yaml
with open("custom_pipeline_config.yaml", "w") as f:
    yaml.dump(asdict(config), f, default_flow_style=False)
```

## 🧪 Testing API

### **Test Suite**

#### **Run Tests**
Execute the automated test suite.

```python
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Run specific test
pytest.main(["tests/test_data_quality.py::TestDataQuality::test_win_percentage_calculation", "-v"])

# Run all tests
pytest.main(["tests/", "-v", "--cov=src"])
```

#### **Custom Tests**
Create custom tests for your use case.

```python
import pytest
import pandas as pd

def test_custom_analysis():
    """Test custom analysis function."""
    data = pd.read_csv("data/models/ncaa_football_ml_dataset.csv")
    
    # Your custom analysis
    result = custom_analysis_function(data)
    
    # Assertions
    assert result is not None
    assert len(result) > 0
    assert 'expected_column' in result.columns

def custom_analysis_function(data):
    """Custom analysis function."""
    # Your analysis logic here
    return data.groupby('conference').agg({
        'win_percentage': 'mean',
        'yards_per_game': 'mean'
    }).reset_index()
```

## 📚 Examples

### **Complete Analysis Workflow**

```python
from src.analytics import AdvancedAnalytics
from src.ml.models import NCAAFootballPredictor
import pandas as pd

# Load data
data = pd.read_csv("data/models/ncaa_football_ml_dataset.csv")

# Initialize analytics
analytics = AdvancedAnalytics(data)

# Perform comprehensive analysis
def comprehensive_team_analysis(team_name, year):
    """Perform comprehensive analysis for a team."""
    
    # Trend analysis
    trend_data = analytics.calculate_trend_analysis(
        team=team_name,
        metric="win_percentage",
        years=[2023, 2024]
    )
    
    # Predictive insights
    predictions_data = analytics.calculate_predictive_insights(
        team=team_name,
        year=year
    )
    
    # Team insights summary
    insights_data = analytics.generate_insights_summary(
        team=team_name,
        year=year
    )
    
    # ML predictions
    predictor = NCAAFootballPredictor(data)
    features_with_targets = predictor.create_targets()
    team_predictions = features_with_targets[features_with_targets['team'] == team_name]
    
    return {
        'trend_analysis': trend_data,
        'predictions': predictions_data,
        'insights': insights_data,
        'ml_predictions': team_predictions
    }

# Analyze Ohio State
ohio_state_analysis = comprehensive_team_analysis("Ohio State", 2025)
print("Ohio State Analysis Complete!")
```

### **Batch Processing**

```python
def analyze_all_teams():
    """Analyze all teams in the dataset."""
    data = pd.read_csv("data/models/ncaa_football_ml_dataset.csv")
    analytics = AdvancedAnalytics(data)
    
    teams = data['team'].unique()
    results = {}
    
    for team in teams:
        try:
            # Quick analysis for each team
            insights = analytics.generate_insights_summary(team, 2024)
            results[team] = {
                'strengths_count': len(insights['strengths']),
                'weaknesses_count': len(insights['weaknesses']),
                'recommendations_count': len(insights['recommendations'])
            }
        except Exception as e:
            results[team] = {'error': str(e)}
    
    return results

# Analyze all teams
all_teams_analysis = analyze_all_teams()
print(f"Analyzed {len(all_teams_analysis)} teams")
```

## 🆘 API Troubleshooting

### **Common Issues**

#### **Import Errors**
```python
# Ensure project root is in Python path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import modules
from src.analytics import AdvancedAnalytics
```

#### **Data Not Found**
```python
# Check if data exists
import os
if not os.path.exists("data/models/ncaa_football_ml_dataset.csv"):
    print("Data not found. Run the data pipeline first.")
    # Run pipeline
    from scripts.run_pipeline import main
    main()
```

#### **API Key Issues**
```python
# Check API key
import os
api_key = os.getenv("CFBD_API_KEY")
if not api_key:
    print("API key not found. Set CFBD_API_KEY environment variable.")
```

---

**🎉 This API documentation provides comprehensive access to all platform capabilities!**

**Ready to build custom applications? Check out our [User Guide](USER_GUIDE.md) for general usage or [Dashboard Tutorials](DASHBOARD_TUTORIALS.md) for detailed walkthroughs.**
