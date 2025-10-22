# 🏈 NCAA Football Analytics Platform

A comprehensive, full-stack data analytics platform for NCAA Division I college football that ingests data from public sources, processes it through machine learning models, and presents insights through interactive dashboards.

## 🌟 Key Features

### 📊 **Interactive Dashboards**
- **Real-time Analytics**: Live filtering by season, conference, and team
- **Performance Metrics**: Win percentages, yards per game, turnover margins
- **Conference Analysis**: Standings and comparative analysis
- **Database Integration**: Fast queries with DuckDB backend
- **Advanced Analytics**: Trend analysis, statistical comparisons, predictive insights

### 🤖 **Machine Learning Predictions**
- **Top 25 Predictions**: Random Forest model with 81.5% accuracy
- **Performance Rankings**: Regression model with 97.7% R² score
- **2025 Season Forecasts**: Predictions for 135+ teams

### 🗄️ **Data Management**
- **Automated Pipeline**: College Football Data API integration
- **Data Processing**: Cleaning, transformation, and feature engineering
- **Database Storage**: DuckDB for fast analytical queries
- **Historical Data**: 2023-2024 seasons with 268 team records

## 🚀 Quick Start

### **Live Dashboard**
The dashboard is deployed and accessible at:
**🔗 [Streamlit Cloud Deployment](https://your-app-name.streamlit.app)**

### **Advanced Analytics Dashboard**
```bash
# Run the advanced analytics dashboard
python scripts/run_analytics_dashboard.py

# Access at: http://localhost:8504
```

**Features:**
- **Trend Analysis**: Historical performance trends and forecasting
- **Performance Clustering**: ML-based team grouping by performance
- **Statistical Comparison**: T-tests and effect size analysis
- **Predictive Insights**: Future performance predictions
- **Conference Analysis**: Conference-wide performance analysis
- **Team Insights**: Comprehensive team analysis and recommendations

### **Production Pipeline**
```bash
# Run pipeline once
python scripts/run_production_pipeline.py run

# Start scheduler (runs daily at 6 AM)
python scripts/run_production_pipeline.py schedule

# Start monitoring dashboard
python scripts/run_production_pipeline.py monitor
```

**Features:**
- **Automated Scheduling**: Daily/weekly/monthly execution
- **Error Handling**: Robust error handling and recovery
- **Monitoring**: Real-time pipeline monitoring and alerts
- **Data Validation**: Automated data quality checks
- **Backup & Recovery**: Automatic data backups
- **Performance Optimization**: Efficient batch processing

### **Local Development**
```bash
# Clone the repository
git clone https://github.com/yourusername/ncaa-football-analytics.git
cd ncaa-football-analytics

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run streamlit_app.py
```

## 📁 Project Structure

```
ncaa-football-analytics/
├── 📊 streamlit_app.py              # Main dashboard (deployment-ready)
├── 📋 requirements.txt              # Python dependencies
├── 📖 README.md                     # This file
├── 🔧 config/
│   └── settings.py                  # Configuration and API keys
├── 📦 src/
│   ├── 🗄️ storage/
│   │   └── simple_database.py       # DuckDB database management
│   ├── 🔄 processing/
│   │   ├── cleaner.py               # Data cleaning and transformation
│   │   └── transformer.py            # Feature engineering
│   ├── 🤖 ml/
│   │   └── models.py                # Machine learning models
│   ├── 📡 ingestion/
│   │   ├── api_client.py            # College Football Data API client
│   │   └── data_collector.py        # Data collection orchestration
│   └── 📈 visualization/
│       ├── enhanced_dashboard.py    # Enhanced dashboard with ML
│       └── streamlit_dashboard.py   # Basic dashboard
├── 🛠️ scripts/
│   ├── run_pipeline.py              # Complete data pipeline
│   ├── setup_database.py            # Database setup
│   ├── train_ml_models.py           # ML model training
│   └── query_database.py            # Database query tool
└── 📊 data/
    ├── raw/                         # Raw API data
    ├── processed/                    # Cleaned data
    └── models/                       # ML models and predictions
```

## 🎯 Dashboard Features

### **📊 Team Performance Analysis**
- **Win Percentage Tracking**: Historical and current season performance
- **Statistical Comparisons**: Yards per game, defensive metrics, turnover margins
- **Team Rankings**: Sortable tables with key performance indicators

### **🏟️ Conference Analysis**
- **Standings**: Conference-specific team rankings
- **Comparative Analysis**: Cross-conference performance metrics
- **Trend Analysis**: Historical conference strength

### **🤖 Machine Learning Insights**
- **Top 25 Predictions**: Probability-based team rankings for 2025
- **Performance Rankings**: Composite scoring system
- **Model Accuracy**: Transparent ML model performance metrics

### **📈 Advanced Analytics**
- **Correlation Analysis**: Performance metric relationships
- **Scatter Plot Analysis**: Multi-dimensional data visualization
- **Interactive Filtering**: Real-time data exploration

## 🔧 Technical Stack

### **Backend**
- **Python 3.9+**: Core programming language
- **DuckDB**: Embedded analytical database
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

### **Machine Learning**
- **Scikit-learn**: Random Forest and Logistic Regression
- **Feature Engineering**: 98+ derived metrics and historical features
- **Model Evaluation**: Accuracy, F1-score, R² metrics

### **Frontend**
- **Streamlit**: Interactive web application framework
- **Plotly**: Advanced data visualization
- **Responsive Design**: Mobile-friendly interface

### **Data Sources**
- **College Football Data API**: Official NCAA statistics
- **Automated Ingestion**: Scheduled data updates
- **Data Validation**: Quality checks and error handling

## 📊 Data Pipeline

### **1. Data Ingestion**
```bash
# Run complete data pipeline
python scripts/run_pipeline.py

# Test API connection
python scripts/test_api.py
```

### **2. Data Processing**
- **Cleaning**: Handle missing values, standardize formats
- **Transformation**: Calculate derived metrics and features
- **Validation**: Quality checks and anomaly detection

### **3. Machine Learning**
```bash
# Train ML models
python scripts/train_ml_models.py

# Generate predictions
python scripts/generate_predictions.py
```

### **4. Database Setup**
```bash
# Setup database
python scripts/setup_database.py

# Query database
python scripts/query_database.py --big-ten 2024
```

## 🎯 Key Metrics & Insights

### **Performance Metrics**
- **Win Percentage**: Calculated from actual game results
- **Yards Per Game**: Offensive efficiency measurement
- **Turnover Margin**: Defensive pressure and ball security
- **Conference Strength**: Comparative conference analysis

### **ML Model Performance**
- **Top 25 Accuracy**: 81.5% prediction accuracy
- **Ranking R² Score**: 97.7% variance explained
- **Feature Importance**: Transparent model interpretability

### **Data Coverage**
- **Teams**: 134+ Division I teams
- **Seasons**: 2023-2024 historical data
- **Conferences**: 11 major conferences
- **Games**: 7,500+ completed games analyzed

## 🚀 Deployment

### **Streamlit Cloud**
The dashboard is deployed on Streamlit Cloud for easy access:
1. **Automatic Updates**: Connected to GitHub repository
2. **Scalable Infrastructure**: Handles multiple concurrent users
3. **Custom Domain**: Professional URL for sharing

### **Local Deployment**
```bash
# Production setup
pip install -r requirements.txt
streamlit run streamlit_app.py --server.port 8501
```

## 🔍 Usage Examples

### **Team Analysis**
```python
# Compare two teams
python scripts/query_database.py --compare Indiana Ohio-State --season 2024

# Get team statistics
python scripts/query_database.py --team Indiana --season 2024
```

### **Conference Analysis**
```bash
# Big Ten standings
python scripts/query_database.py --big-ten 2024

# Top teams
python scripts/query_database.py --top-teams 2024 --limit 10
```

### **Custom Queries**
```sql
-- Custom SQL analysis
python scripts/query_database.py --sql "
SELECT team, win_percentage, yards_per_game 
FROM ncaa_football_data 
WHERE year = 2024 
ORDER BY win_percentage DESC 
LIMIT 10"
```

## 📈 Performance

### **Database Performance**
- **Query Speed**: Sub-second response times
- **Scalability**: Handles 268+ records efficiently
- **Memory Usage**: Optimized for analytical workloads

### **Dashboard Performance**
- **Load Time**: < 3 seconds initial load
- **Filtering**: Real-time data updates
- **Caching**: Streamlit caching for optimal performance

## 🤝 Contributing

### **Development Setup**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### **Code Standards**
- **Python**: PEP 8 style guidelines
- **Documentation**: Docstrings for all functions
- **Testing**: Unit tests for critical functions
- **Type Hints**: Type annotations for better code clarity

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **College Football Data API**: For providing comprehensive NCAA statistics
- **Streamlit**: For the excellent web application framework
- **DuckDB**: For fast analytical database capabilities
- **Plotly**: For beautiful, interactive visualizations

## 📞 Support

For questions, issues, or contributions:
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/ncaa-football-analytics/issues)
- **Documentation**: Check the [Wiki](https://github.com/yourusername/ncaa-football-analytics/wiki)
- **Discussions**: Join the [Discussions](https://github.com/yourusername/ncaa-football-analytics/discussions)

---

**🏈 Built with ❤️ for college football fans and data enthusiasts**

*Last updated: October 2024*