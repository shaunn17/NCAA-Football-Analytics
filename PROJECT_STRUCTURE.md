# 📁 NCAA Football Analytics - Project Structure

## 🏗️ Improved Organization

This document outlines the improved project structure for better organization and maintainability.

## 📂 Current Structure

```
ncaa-football-analytics/
├── 📊 dashboard/                          # Main dashboard applications
│   ├── streamlit_app.py                  # Production dashboard (main entry point)
│   └── simple_dashboard.py                # Simplified dashboard (fallback)
│
├── 📚 docs/                                # Documentation
│   ├── README.md                          # Documentation overview
│   ├── USER_GUIDE.md                      # Getting started guide
│   ├── DASHBOARD_TUTORIALS.md             # Dashboard tutorials
│   ├── ADVANCED_ANALYTICS.md              # Advanced analytics guide
│   ├── API_DOCUMENTATION.md               # API reference
│   ├── PRODUCTION_PIPELINE.md             # Production pipeline guide
│   ├── FAQ_TROUBLESHOOTING.md            # Common issues & solutions
│   ├── DOCUMENTATION_INDEX.md            # Complete documentation index
│   └── external/                          # External documentation
│       ├── DEPLOYMENT.md                   # Deployment guide
│       ├── DATABASE_INTEGRATION.md        # Database setup guide
│       └── QUICKSTART.md                   # Quick start guide
│
├── 📊 data/                                # Data files
│   ├── raw/                               # Raw API data
│   │   ├── *.json                         # JSON data files
│   │   └── *.csv                          # CSV data files
│   ├── processed/                         # Cleaned data
│   │   └── *_clean.csv                    # Processed datasets
│   ├── models/                             # ML models and predictions
│   │   ├── 2025_predictions.csv           # ML predictions
│   │   ├── ncaa_football_ml_dataset.csv   # ML-ready dataset
│   │   ├── *.joblib                       # Trained models
│   │   └── *.pkl                          # Pickled models
│   ├── *.duckdb                            # DuckDB database files
│   └── pipeline_state.db                   # Pipeline state database
│
├── 📁 src/                                  # Source code
│   ├── analytics/                          # Advanced analytics
│   │   ├── advanced_analytics.py         # Analytics engine
│   │   └── visualizations.py              # Visualization functions
│   ├── ingestion/                          # Data collection
│   │   ├── api_client.py                  # API client
│   │   └── data_collector.py              # Data collector
│   ├── processing/                         # Data processing
│   │   ├── cleaner.py                     # Data cleaning
│   │   └── transformer.py                 # Feature engineering
│   ├── ml/                                 # Machine learning
│   │   └── models.py                      # ML models
│   ├── storage/                            # Database management
│   │   ├── database.py                    # PostgreSQL database
│   │   └── simple_database.py             # DuckDB database
│   ├── visualization/                       # Visualization components
│   │   ├── analytics_dashboard.py        # Analytics dashboard
│   │   ├── dashboard.py                   # Basic dashboard
│   │   ├── enhanced_dashboard.py          # Enhanced dashboard
│   │   ├── streamlit_dashboard.py         # Streamlit components
│   │   └── pipeline_monitor.py            # Pipeline monitoring
│   ├── pipeline/                            # Production pipeline
│   │   └── production_pipeline.py        # Automated pipeline
│   └── utils/                               # Utility functions
│
├── 🔧 config/                              # Configuration
│   ├── settings.py                         # Application settings
│   └── pipeline_config.yaml               # Pipeline configuration
│
├── 🛠️ scripts/                             # Utility scripts
│   ├── run_pipeline.py                    # Data pipeline runner
│   ├── run_production_pipeline.py         # Production pipeline
│   ├── run_dashboard.py                    # Dashboard launcher
│   ├── run_analytics_dashboard.py         # Analytics dashboard launcher
│   ├── train_ml_models.py                 # ML model training
│   ├── train_simple_ml.py                 # Simple ML training
│   ├── setup_database.py                  # Database setup
│   ├── setup_simple_database.py           # DuckDB setup
│   ├── query_database.py                  # Database query tool
│   ├── benchmark_database.py              # Database benchmarking
│   ├── test_api.py                        # API testing
│   ├── run_tests.py                       # Test runner
│   ├── check_api_usage.py                 # API usage checker
│   ├── api_usage_monitor.py               # API usage monitor
│   └── fix_win_percentages.py            # Data fix utility
│
├── 🧪 tests/                                # Test suite
│   ├── conftest.py                        # Pytest configuration
│   ├── test_api_connections.py            # API tests
│   ├── test_data_quality.py               # Data quality tests
│   ├── test_database.py                   # Database tests
│   ├── test_integration.py                # Integration tests
│   └── test_ml_models.py                  # ML model tests
│
├── 📝 logs/                                # Log files
│   ├── pipeline_*.log                     # Pipeline logs
│   └── old/                               # Old log files
│
├── 📓 notebooks/                            # Jupyter notebooks
│
├── 🚀 deployment/                           # Deployment configuration
│   └── {github_actions}/                  # GitHub Actions workflows
│
├── 📦 Root Files                           # Main project files
│   ├── README.md                          # Project overview
│   ├── requirements.txt                   # Python dependencies
│   ├── setup.py                           # Package setup
│   ├── pytest.ini                         # Pytest configuration
│   ├── env.example                        # Environment template
│   ├── .gitignore                         # Git ignore rules
│   └── PROJECT_STRUCTURE.md               # This file
│
└── 🔐 venv/                                # Virtual environment (not tracked)
```

## 🎯 Key Improvements

### ✅ **1. Organized Log Files**
- Moved all `pipeline_*.log` files to `logs/old/`
- Centralized logging in `logs/` directory
- Updated `.gitignore` to handle old logs

### ✅ **2. Consolidated Documentation**
- All documentation in `docs/` directory
- External docs (DEPLOYMENT, DATABASE_INTEGRATION) in `docs/external/`
- Clear separation between guides and references

### ✅ **3. Grouped Dashboard Files**
- Created `dashboard/` directory for main dashboard applications
- Separated production (`streamlit_app.py`) from simple (`simple_dashboard.py`) dashboards
- Better organization for multiple dashboard variants

### ✅ **4. Clean Root Directory**
- Root only contains essential project files
- Configuration, documentation, and data properly organized
- Easier navigation and maintenance

## 📊 Benefits

### **For Development**
- 🔍 **Easier Navigation** - Related files grouped together
- 🧹 **Cleaner Root** - No clutter at project root
- 📚 **Better Organization** - Logical grouping of components
- 🛠️ **Easier Maintenance** - Clear structure for updates

### **For Collaboration**
- 📖 **Clear Documentation** - All docs in one place
- 🎯 **Better Discovery** - Easy to find what you need
- 📝 **Consistent Structure** - Standard project layout
- 🤝 **Team Friendly** - Intuitive organization

### **For Deployment**
- 🚀 **Production Ready** - Clean structure for deployment
- 📦 **Easy Packaging** - Clear separation of concerns
- 🔧 **Simple Configuration** - Config files in dedicated directory
- 📊 **Better Monitoring** - Centralized logs

## 🎯 Usage Guide

### **Running Dashboards**
```bash
# Main production dashboard
streamlit run dashboard/streamlit_app.py

# Simple dashboard (fallback)
streamlit run dashboard/simple_dashboard.py

# Analytics dashboard
python scripts/run_analytics_dashboard.py
```

### **Running Scripts**
```bash
# Data pipeline
python scripts/run_pipeline.py

# Production pipeline
python scripts/run_production_pipeline.py

# Train ML models
python scripts/train_ml_models.py
```

### **Testing**
```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/test_ml_models.py
```

### **Documentation**
All documentation is now in the `docs/` directory:
- Start with `README.md` for overview
- Use `docs/USER_GUIDE.md` for getting started
- See `docs/DOCUMENTATION_INDEX.md` for complete index

## 🔄 Migration Notes

If you have existing scripts or documentation that references the old structure, update them to use the new paths:

**Old Path** → **New Path**
- `streamlit_app.py` → `dashboard/streamlit_app.py`
- `simple_dashboard.py` → `dashboard/simple_dashboard.py`
- `DATABASE_INTEGRATION.md` → `docs/external/DATABASE_INTEGRATION.md`
- `DEPLOYMENT.md` → `docs/external/DEPLOYMENT.md`
- `QUICKSTART.md` → `docs/external/QUICKSTART.md`
- `pipeline_*.log` → `logs/old/pipeline_*.log`

## 📈 Next Steps

Consider these further improvements:
1. **Add more notebooks** to `notebooks/` directory for analysis
2. **Create deployment configs** in `deployment/`
3. **Add CI/CD workflows** in `deployment/{github_actions}/`
4. **Expand test coverage** in `tests/`
5. **Add type hints** throughout codebase

---

**Last Updated:** October 2024
**Structure Version:** 2.0
