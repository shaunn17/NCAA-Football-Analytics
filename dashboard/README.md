# 📊 Dashboard Applications

This directory contains the main dashboard applications for the NCAA Football Analytics platform.

## 🎯 Available Dashboards

### **1. streamlit_app.py**
**Production Dashboard** - The main dashboard with full features including ML predictions, database integration, and advanced analytics.

**Run:**
```bash
streamlit run dashboard/streamlit_app.py
```

**Features:**
- ✅ ML-powered Top 25 predictions
- ✅ Performance rankings
- ✅ Team and conference analysis
- ✅ Interactive visualizations
- ✅ Database integration with fallback
- ✅ Production-ready deployment

---

### **2. simple_dashboard.py**
**Simple Dashboard** - Simplified dashboard that works with CSV files only, no database required.

**Run:**
```bash
streamlit run dashboard/simple_dashboard.py
```

**Features:**
- ✅ No database dependencies
- ✅ Works with CSV files only
- ✅ Basic ML predictions
- ✅ Team statistics
- ✅ Good for quick testing or demos
- ✅ Easy deployment

---

## 🚀 Quick Start

### **Option 1: Run Production Dashboard**
```bash
# From project root
streamlit run dashboard/streamlit_app.py
```

Access at: `http://localhost:8501`

### **Option 2: Run Simple Dashboard**
```bash
# From project root
streamlit run dashboard/simple_dashboard.py
```

Access at: `http://localhost:8501`

### **Option 3: Run Analytics Dashboard**
```bash
# From project root
python scripts/run_analytics_dashboard.py
```

Access at: `http://localhost:8504`

---

## 📋 Prerequisites

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Environment:**
   ```bash
   cp env.example .env
   # Add your API key to .env
   ```

3. **Run Data Pipeline** (Optional):
   ```bash
   python scripts/run_pipeline.py
   ```

---

## 🎨 Dashboard Features

### **Common Features (Both Dashboards):**
- 📊 Team performance analysis
- 🏟️ Conference standings
- 📈 Statistical visualizations
- 🤖 ML predictions
- 🔍 Interactive filtering
- 📱 Responsive design

### **Production Dashboard Only:**
- 🗄️ Database integration
- ⚡ Faster queries with caching
- 📊 Enhanced ML predictions
- 🔄 Real-time data updates
- 💾 State management

---

## 🔧 Configuration

Dashboard settings can be configured in:
- `config/settings.py` - General settings
- `.env` - Environment variables (API keys, etc.)

---

## 📚 Documentation

For more information, see:
- [User Guide](../docs/USER_GUIDE.md)
- [Dashboard Tutorials](../docs/DASHBOARD_TUTORIALS.md)
- [Project Structure](../PROJECT_STRUCTURE.md)

---

## 🐛 Troubleshooting

### **Dashboard won't start:**
```bash
# Check if port is in use
lsof -i :8501

# Kill existing process
pkill -f streamlit
```

### **Data not loading:**
```bash
# Run the data pipeline
python scripts/run_pipeline.py
```

### **Import errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

---

**Last Updated:** October 2024
