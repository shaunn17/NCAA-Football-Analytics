# ❓ FAQ & Troubleshooting

This comprehensive guide answers common questions and provides solutions to typical issues you might encounter with the NCAA Football Analytics Platform.

## 🚀 Getting Started FAQ

### **Q: How do I get started with the platform?**
**A**: The easiest way is to use our live dashboard at [your-app-name.streamlit.app](https://your-app-name.streamlit.app). For local installation, follow our [Getting Started Guide](USER_GUIDE.md).

### **Q: Do I need to install anything to use the live dashboard?**
**A**: No! The live dashboard runs in your web browser and requires no installation.

### **Q: What data is available in the platform?**
**A**: We provide data for 134+ Division I teams across 11 conferences for the 2023-2024 seasons, including:
- Team statistics (win percentage, yards per game, etc.)
- Conference standings
- Machine learning predictions
- Historical performance data

### **Q: How often is the data updated?**
**A**: Data is updated regularly through our automated pipeline. The live dashboard reflects the most current data available.

### **Q: Is there an API available?**
**A**: Yes! The platform includes programmatic access through our data processing modules. Check our [API Documentation](API_DOCUMENTATION.md) for details.

## 📊 Dashboard FAQ

### **Q: Why is the dashboard loading slowly?**
**A**: This can happen for several reasons:
- **Large dataset**: Try using filters to reduce data size
- **System resources**: Close other applications to free up memory
- **Network issues**: Check your internet connection
- **Database not set up**: Run `python scripts/setup_simple_database.py`

**Solution**:
```bash
# Set up database for faster queries
python scripts/setup_simple_database.py

# Use filters in the dashboard
# Select specific teams or conferences
```

### **Q: Why am I seeing "No data available" messages?**
**A**: This usually means the data pipeline hasn't been run yet.

**Solution**:
```bash
# Run the data pipeline
python scripts/run_pipeline.py

# Or run individual components
python scripts/test_api.py
python scripts/train_ml_models.py
```

### **Q: The charts aren't displaying properly. What should I do?**
**A**: This could be due to browser compatibility or data issues.

**Solutions**:
1. **Try a different browser** (Chrome, Firefox, Safari)
2. **Clear browser cache** and refresh
3. **Check data availability**:
   ```bash
   ls data/models/
   ```
4. **Restart the dashboard**:
   ```bash
   python scripts/run_dashboard.py enhanced
   ```

### **Q: Why are some team names missing or incorrect?**
**A**: This could be due to data quality issues or API changes.

**Solutions**:
1. **Check data quality**:
   ```bash
   python scripts/run_pipeline.py
   ```
2. **Verify team names** in the raw data
3. **Report the issue** on GitHub

### **Q: How do I compare more than two teams?**
**A**: The current dashboard supports two-team comparisons. For multiple teams:

**Solutions**:
1. **Use the Advanced Analytics Dashboard** for statistical comparisons
2. **Export data** and use external tools
3. **Run multiple comparisons** sequentially

### **Q: Why are the ML predictions different from what I expect?**
**A**: ML predictions are based on historical data and statistical patterns.

**Factors affecting predictions**:
- **Data quality**: Better data = better predictions
- **Historical patterns**: Predictions based on past performance
- **Model limitations**: No model is 100% accurate
- **External factors**: Injuries, coaching changes, etc.

**To improve predictions**:
- **Use more recent data**
- **Consider additional factors**
- **Understand model limitations**

## 🔬 Advanced Analytics FAQ

### **Q: Why is the Advanced Analytics Dashboard not loading?**
**A**: This requires additional dependencies.

**Solution**:
```bash
# Install required dependencies
pip install scipy scikit-learn

# Run the advanced analytics dashboard
python scripts/run_analytics_dashboard.py
```

### **Q: Why am I getting "Insufficient data" errors?**
**A**: Advanced analytics require sufficient historical data.

**Solutions**:
1. **Check data availability**:
   ```bash
   ls data/models/
   ```
2. **Run data pipeline**:
   ```bash
   python scripts/run_pipeline.py
   ```
3. **Use different time ranges** or teams

### **Q: Why are trend analysis results showing "Declining" when the team is good?**
**A**: Trend analysis looks at year-over-year changes, not absolute performance.

**Explanation**:
- **Declining trend**: Performance getting worse over time
- **Good team**: Can still be good but declining
- **Context matters**: Consider the overall performance level

**Example**: A team with 90% win rate declining to 85% is still excellent but trending down.

### **Q: Why are statistical comparisons showing "Not Significant" for obvious differences?**
**A**: Statistical significance depends on sample size and variance.

**Factors**:
- **Sample size**: Small samples = less statistical power
- **Variance**: High variance = less significant differences
- **Effect size**: Small differences = less significant

**Solutions**:
1. **Use larger datasets**
2. **Consider practical significance**
3. **Look at effect sizes**

### **Q: Why are clustering results different from what I expect?**
**A**: Clustering is based on statistical similarity, not subjective assessment.

**Factors**:
- **Metrics used**: Different metrics = different clusters
- **Algorithm**: K-means creates statistical groups
- **Data quality**: Better data = better clusters

**Solutions**:
1. **Adjust metrics** used for clustering
2. **Check data quality**
3. **Understand clustering algorithm**

## 🏭 Production Pipeline FAQ

### **Q: How do I set up the production pipeline?**
**A**: Follow these steps:

**Setup**:
```bash
# Install dependencies
pip install schedule pyyaml psutil

# Configure pipeline
# Edit config/pipeline_config.yaml

# Run pipeline once
python scripts/run_production_pipeline.py run

# Start scheduler
python scripts/run_production_pipeline.py schedule
```

### **Q: Why is the pipeline failing?**
**A**: Common causes and solutions:

**API Issues**:
```bash
# Check API key
echo $CFBD_API_KEY

# Test API connection
python scripts/test_api.py
```

**Data Issues**:
```bash
# Check data quality
python scripts/run_pipeline.py

# Validate data
python -c "import pandas as pd; df = pd.read_csv('data/models/ncaa_football_ml_dataset.csv'); print(df.info())"
```

**System Issues**:
```bash
# Check system resources
python scripts/run_production_pipeline.py monitor

# Check logs
tail -f logs/pipeline_$(date +%Y%m%d).log
```

### **Q: How do I monitor the pipeline?**
**A**: Use the monitoring dashboard:

```bash
# Start monitoring dashboard
python scripts/run_production_pipeline.py monitor

# Access at: http://localhost:8505
```

**Monitor**:
- **Pipeline status**
- **Performance metrics**
- **Error tracking**
- **System resources**

### **Q: Why is the pipeline running slowly?**
**A**: Performance issues can have several causes:

**Solutions**:
1. **Check system resources**:
   ```bash
   python scripts/run_production_pipeline.py monitor
   ```
2. **Optimize configuration**:
   ```yaml
   # config/pipeline_config.yaml
   batch_size: 500  # Increase batch size
   max_retries: 3   # Reduce retries
   ```
3. **Use database** for faster queries
4. **Close other applications**

### **Q: How do I configure notifications?**
**A**: Edit the configuration file:

```yaml
# config/pipeline_config.yaml
enable_notifications: true
notification_email: "your-email@example.com"
slack_webhook: "https://hooks.slack.com/services/..."
```

**Setup**:
1. **Email**: Configure SMTP settings
2. **Slack**: Create webhook URL
3. **Test notifications**: Run pipeline and check

## 🛠️ Technical Troubleshooting

### **Installation Issues**

#### **Problem**: `ModuleNotFoundError: No module named 'src'`
**Solution**:
```bash
# Ensure you're in the project root
cd /path/to/ncaa-football-analytics

# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or use the scripts
python scripts/run_dashboard.py enhanced
```

#### **Problem**: `PermissionError` when installing packages
**Solution**:
```bash
# Use virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

#### **Problem**: `Port already in use` error
**Solution**:
```bash
# Find process using port
lsof -i :8502

# Kill the process
kill <PID>

# Or use different port
python scripts/run_dashboard.py enhanced --port 8503
```

### **Data Issues**

#### **Problem**: Data files are missing
**Solution**:
```bash
# Check if data exists
ls data/models/

# Run data pipeline
python scripts/run_pipeline.py

# Check individual components
python scripts/test_api.py
python scripts/train_ml_models.py
```

#### **Problem**: Data quality issues
**Solution**:
```bash
# Check data quality
python -c "
import pandas as pd
df = pd.read_csv('data/models/ncaa_football_ml_dataset.csv')
print('Data shape:', df.shape)
print('Null values:', df.isnull().sum().sum())
print('Duplicate rows:', df.duplicated().sum())
"

# Fix data quality
python scripts/fix_win_percentages.py
python scripts/run_pipeline.py
```

#### **Problem**: API connection issues
**Solution**:
```bash
# Check API key
echo $CFBD_API_KEY

# Test API connection
python scripts/test_api.py

# Check network connectivity
ping api.collegefootballdata.com
```

### **Performance Issues**

#### **Problem**: Dashboard is slow
**Solution**:
```bash
# Set up database for faster queries
python scripts/setup_simple_database.py

# Use filters to reduce data size
# Close other applications
# Check system resources
```

#### **Problem**: Memory usage is high
**Solution**:
```bash
# Monitor memory usage
python scripts/run_production_pipeline.py monitor

# Reduce batch sizes
# Use data filters
# Close other applications
```

#### **Problem**: CPU usage is high
**Solution**:
```bash
# Check CPU usage
python scripts/run_production_pipeline.py monitor

# Reduce parallel processing
# Optimize queries
# Use caching
```

### **Database Issues**

#### **Problem**: Database connection errors
**Solution**:
```bash
# Check database file
ls data/ncaa_football_simple.duckdb

# Recreate database
rm data/ncaa_football_simple.duckdb
python scripts/setup_simple_database.py
```

#### **Problem**: Database queries are slow
**Solution**:
```bash
# Check database performance
python scripts/benchmark_database.py

# Optimize queries
# Use indexes
# Reduce data size
```

### **ML Model Issues**

#### **Problem**: ML models are not training
**Solution**:
```bash
# Check data availability
python -c "
import pandas as pd
df = pd.read_csv('data/models/ncaa_football_ml_dataset.csv')
print('Data shape:', df.shape)
print('Columns:', df.columns.tolist())
"

# Train models
python scripts/train_ml_models.py

# Check model files
ls data/models/*.joblib
```

#### **Problem**: Predictions are inaccurate
**Solution**:
```bash
# Check model performance
python scripts/train_ml_models.py

# Validate data quality
python scripts/run_pipeline.py

# Retrain models with better data
```

## 🔧 Configuration Issues

### **Environment Variables**

#### **Problem**: API key not found
**Solution**:
```bash
# Set environment variable
export CFBD_API_KEY="your_api_key_here"

# Or create .env file
echo "CFBD_API_KEY=your_api_key_here" > .env
```

#### **Problem**: Configuration not loading
**Solution**:
```bash
# Check configuration file
cat config/settings.py

# Check environment variables
env | grep CFBD

# Test configuration
python -c "from config.settings import settings; print(settings.cfbd_api_key)"
```

### **File Permissions**

#### **Problem**: Cannot write to data directory
**Solution**:
```bash
# Check permissions
ls -la data/

# Fix permissions
chmod 755 data/
chmod 644 data/*

# Or run as different user
sudo chown -R $USER:$USER data/
```

## 📞 Getting Help

### **Self-Help Resources**

1. **Documentation**: Check all guide files
2. **Logs**: Review error logs for details
3. **GitHub Issues**: Search existing issues
4. **Community**: Join community discussions

### **Reporting Issues**

When reporting issues, include:

1. **Error Message**: Complete error text
2. **Steps to Reproduce**: What you were doing
3. **Environment**: OS, Python version, etc.
4. **Logs**: Relevant log files
5. **Screenshots**: If applicable

### **Contact Information**

- **GitHub Issues**: [Create an issue](https://github.com/yourusername/ncaa-football-analytics/issues)
- **Email Support**: support@example.com
- **Community Discord**: [Join our server](https://discord.gg/example)
- **Twitter**: [@NCAAFootballAnalytics](https://twitter.com/example)

## 🎯 Best Practices

### **Prevention**

1. **Regular Updates**: Keep dependencies updated
2. **Backup Data**: Regular data backups
3. **Monitor Resources**: Watch system usage
4. **Test Changes**: Test before deploying
5. **Document Issues**: Keep track of problems

### **When Things Go Wrong**

1. **Check Logs**: Always check logs first
2. **Restart Services**: Try restarting
3. **Check Resources**: Monitor system usage
4. **Isolate Issues**: Narrow down the problem
5. **Seek Help**: Don't hesitate to ask

### **Maintenance**

1. **Regular Cleanup**: Clean old logs and backups
2. **Monitor Performance**: Watch for degradation
3. **Update Dependencies**: Keep packages current
4. **Test Functionality**: Regular testing
5. **Document Changes**: Keep track of modifications

---

**🎉 This FAQ should help you resolve most common issues!**

**Still need help? Check out our [User Guide](USER_GUIDE.md) for general usage or [Dashboard Tutorials](DASHBOARD_TUTORIALS.md) for detailed walkthroughs.**
