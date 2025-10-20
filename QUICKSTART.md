# 🏈 NCAA Football Analytics - Quick Start Guide

This guide will help you get the NCAA Football Analytics platform up and running quickly.

## 🚀 Prerequisites

- Python 3.9 or higher
- College Football Data API key (free at [collegefootballdata.com](https://collegefootballdata.com))
- Git (optional, for version control)

## 📋 Installation Steps

### 1. Clone and Setup

```bash
# Navigate to your project directory
cd "/Users/shaunfigueiro/Desktop/Projects/NCAA Football Analytics"

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### 2. Configure API Key

Create a `.env` file in the project root:

```bash
# Copy the example file
cp env.example .env

# Edit the .env file and add your API key
COLLEGE_FOOTBALL_DATA_API_KEY=your_api_key_here
```

**Get your free API key:**
1. Visit [collegefootballdata.com](https://collegefootballdata.com)
2. Sign up for a free account
3. Generate an API key
4. Add it to your `.env` file

### 3. Test API Connection

```bash
# Test the API connection
python scripts/test_api.py
```

Expected output:
```
🏈 NCAA Football Analytics - API Test
==================================================
🔍 Testing API connection...
✅ API connection successful! Found 11 conferences
  - ACC (ACC)
  - Big 12 (B12)
  - Big Ten (B1G)
  - SEC (SEC)
  - Pac-12 (P12)
  ...

📊 Testing data collection...
Collecting conferences...
✅ Collected 11 conferences
Collecting teams...
✅ Collected 130 teams
Collecting team stats for 2023...
✅ Collected stats for 130 teams

🎉 All tests passed! API is ready for data collection.
```

## 📊 Running the Data Pipeline

### Option 1: Complete Pipeline (Recommended)

```bash
# Run the complete pipeline (collection + cleaning + transformation)
python scripts/run_pipeline.py
```

This will:
- Collect data from the API for seasons 2018-2024
- Clean and standardize the data
- Create ML-ready features
- Save processed datasets

### Option 2: Step-by-Step Pipeline

```bash
# Step 1: Collect data only
python scripts/run_pipeline.py --skip-cleaning --skip-transformation

# Step 2: Clean data only
python scripts/run_pipeline.py --skip-collection --skip-transformation

# Step 3: Transform data only
python scripts/run_pipeline.py --skip-collection --skip-cleaning
```

### Option 3: Custom Parameters

```bash
# Collect data for specific seasons
python scripts/run_pipeline.py --seasons 2022 2023 2024

# Focus on specific conferences
python scripts/run_pipeline.py --conferences "Big Ten" "SEC" "ACC"

# Use custom API key
python scripts/run_pipeline.py --api-key your_api_key_here
```

## 📁 Understanding the Output

After running the pipeline, you'll find:

```
data/
├── raw/                    # Raw API data (JSON and CSV)
│   ├── conferences.json
│   ├── teams_all.csv
│   ├── team_stats_2023.csv
│   └── games_2023.csv
├── processed/              # Cleaned data
│   ├── conferences_clean.csv
│   ├── teams_clean.csv
│   ├── team_stats_2023_clean.csv
│   └── games_2023_clean.csv
└── models/                 # ML-ready datasets
    └── ncaa_football_ml_dataset.csv
```

## 🔧 Configuration

### Key Settings (config/settings.py)

```python
# Seasons to collect (default: 2018-2024)
seasons_to_collect = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

# Focus conferences
conferences_to_focus = [
    "Big Ten", "SEC", "ACC", "Big 12", "Pac-12",
    "American Athletic", "Mountain West", "MAC", "Sun Belt"
]

# API rate limiting
api_rate_limit = 100  # requests per minute
```

### Environment Variables (.env)

```bash
# Required
COLLEGE_FOOTBALL_DATA_API_KEY=your_api_key_here

# Optional
DATABASE_URL=postgresql://localhost:5432/ncaa_football
USE_DUCKDB=false
LOG_LEVEL=INFO
```

## 📈 What's Next?

Once you have data collected and processed:

1. **Explore the Data**: Open `data/models/ncaa_football_ml_dataset.csv` in Excel or a Jupyter notebook
2. **Build Visualizations**: Use the dashboard modules (coming next)
3. **Train ML Models**: Use the machine learning modules (coming next)
4. **Deploy Dashboard**: Deploy to Streamlit Cloud or Render

## 🆘 Troubleshooting

### Common Issues

**API Key Error:**
```
❌ No API key found!
Please set COLLEGE_FOOTBALL_DATA_API_KEY in your environment or .env file
```
- Make sure you have a `.env` file with your API key
- Verify the API key is correct at collegefootballdata.com

**Rate Limiting:**
```
Rate limit reached. Sleeping for X seconds
```
- This is normal - the API has rate limits
- The pipeline will automatically handle this

**Missing Dependencies:**
```
ModuleNotFoundError: No module named 'requests'
```
- Run: `pip install -r requirements.txt`

**Permission Errors:**
```
PermissionError: [Errno 13] Permission denied
```
- Make sure you have write permissions in the project directory
- On Windows, try running as administrator

### Getting Help

1. Check the logs in `pipeline_YYYYMMDD_HHMMSS.log`
2. Run with debug logging: `python scripts/run_pipeline.py --log-level DEBUG`
3. Test individual components: `python scripts/test_api.py`

## 📊 Expected Data Volume

For a typical run (2018-2024, all conferences):
- **Teams**: ~130 teams
- **Games**: ~8,000-10,000 games
- **Team Stats**: ~910 team-season records
- **Processing Time**: 10-30 minutes (depending on API rate limits)

## 🎯 Success Indicators

You'll know the pipeline worked when you see:
- ✅ All test steps pass
- 📁 Data files created in `data/raw/` and `data/processed/`
- 📊 ML dataset created in `data/models/`
- 🎉 "Pipeline completed successfully!" message

Ready to start? Run: `python scripts/test_api.py`


