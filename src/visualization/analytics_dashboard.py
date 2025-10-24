"""
Enhanced Analytics Dashboard for NCAA Football Analytics Platform

This dashboard provides advanced analytics capabilities including:
- Trend analysis and forecasting
- Statistical analysis and metrics
- Comparative analytics
- Performance insights
- Predictive modeling
- Interactive visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import analytics modules
try:
    from src.analytics import AdvancedAnalytics, AdvancedVisualizations
    from src.storage.simple_database import SimpleDatabaseManager
    ANALYTICS_AVAILABLE = True
except ImportError as e:
    ANALYTICS_AVAILABLE = False
    logger.warning(f"Analytics modules not available: {e}")

def load_data():
    """Load data from database or CSV fallback."""
    if ANALYTICS_AVAILABLE:
        try:
            db = SimpleDatabaseManager()
            # Get all data for analytics
            query = "SELECT * FROM ncaa_football_data"
            return db.query(query)
        except Exception as e:
            logger.warning(f"Database query failed: {e}")
    
    # Fallback to CSV
    csv_path = Path("data/models/ncaa_football_ml_dataset.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        st.error("No data available. Please run the data pipeline first.")
        return None

def main():
    """Main analytics dashboard."""
    st.set_page_config(
        page_title="NCAA Football Advanced Analytics",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🏈 NCAA Football Advanced Analytics")
    st.markdown("---")
    
    # Check if analytics are available
    if not ANALYTICS_AVAILABLE:
        st.error("⚠️ Advanced analytics modules are not available. Please check your installation.")
        return
    
    # Load data
    with st.spinner("Loading data..."):
        data = load_data()
    
    if data is None or data.empty:
        st.error("No data available. Please run the data pipeline first.")
        return
    
    # Initialize analytics engine
    analytics = AdvancedAnalytics(data)
    
    # Sidebar controls
    st.sidebar.header("🎛️ Analytics Controls")
    
    # Get available teams and years (handle mixed data types)
    teams = sorted([str(t) for t in data['team'].dropna().unique()])
    years = sorted([int(y) for y in data['year'].dropna().unique()])
    conferences = sorted([str(c) for c in data['conference'].dropna().unique()])
    
    # Analysis type selection
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Trend Analysis", "Performance Clustering", "Statistical Comparison", 
         "Predictive Insights", "Conference Analysis", "Team Insights Summary"]
    )
    
    st.sidebar.markdown("---")
    
    # Main content area
    if analysis_type == "Trend Analysis":
        render_trend_analysis(analytics, teams, years)
    elif analysis_type == "Performance Clustering":
        render_performance_clustering(analytics, years)
    elif analysis_type == "Statistical Comparison":
        render_statistical_comparison(analytics, teams, years)
    elif analysis_type == "Predictive Insights":
        render_predictive_insights(analytics, teams, years)
    elif analysis_type == "Conference Analysis":
        render_conference_analysis(analytics, conferences, years)
    elif analysis_type == "Team Insights Summary":
        render_team_insights_summary(analytics, teams, years)

def render_trend_analysis(analytics: AdvancedAnalytics, teams: List[str], years: List[int]):
    """Render trend analysis section."""
    st.header("📈 Trend Analysis")
    st.markdown("Analyze historical trends and performance patterns for teams.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_team = st.selectbox("Select Team", teams)
    
    with col2:
        metrics = ['win_percentage', 'yards_per_game', 'yards_allowed_per_game', 
                  'turnover_margin', 'offensive_efficiency', 'defensive_efficiency']
        selected_metric = st.selectbox("Select Metric", metrics)
    
    with col3:
        selected_years = st.multiselect("Select Years", years, default=years[-3:])
    
    if st.button("Analyze Trends", type="primary"):
        with st.spinner("Analyzing trends..."):
            trend_data = analytics.calculate_trend_analysis(
                selected_team, selected_metric, selected_years
            )
        
        if "error" not in trend_data:
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Trend Chart")
                fig = AdvancedVisualizations.create_trend_analysis_chart(trend_data)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📋 Trend Summary")
                st.metric("Trend Direction", trend_data['trend_direction'])
                st.metric("R² Score", f"{trend_data['r_squared']:.3f}")
                st.metric("Average YoY Change", f"{trend_data['avg_yoy_change']:.3f}")
                st.metric("Volatility", f"{trend_data['volatility']:.3f}")
                st.metric("Momentum", f"{trend_data['momentum']:.3f}")
                
                if trend_data['current_value'] is not None:
                    st.metric("Current Value", f"{trend_data['current_value']:.3f}")
                
                st.info(f"Best Year: {int(trend_data['best_year'])} | Worst Year: {int(trend_data['worst_year'])}")
        else:
            st.error(trend_data['error'])

def render_performance_clustering(analytics: AdvancedAnalytics, years: List[int]):
    """Render performance clustering section."""
    st.header("🎯 Performance Clustering")
    st.markdown("Group teams by performance characteristics using machine learning clustering.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_year = st.selectbox("Select Year", years)
    
    with col2:
        metrics = ['win_percentage', 'yards_per_game', 'yards_allowed_per_game', 
                  'turnover_margin', 'offensive_efficiency', 'defensive_efficiency']
        selected_metrics = st.multiselect("Select Metrics", metrics, default=metrics[:4])
    
    if st.button("Perform Clustering", type="primary"):
        with st.spinner("Performing clustering analysis..."):
            cluster_data = analytics.calculate_performance_clusters(selected_year, selected_metrics)
        
        if "error" not in cluster_data:
            # Display results
            st.subheader("📊 Clustering Results")
            fig = AdvancedVisualizations.create_performance_cluster_chart(cluster_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display cluster details
            st.subheader("📋 Cluster Details")
            clusters = cluster_data['clusters']
            
            for cluster_name, cluster_info in clusters.items():
                with st.expander(f"{cluster_name} ({cluster_info['count']} teams)"):
                    st.write(f"**Teams:** {', '.join(cluster_info['teams'])}")
                    st.write(f"**Average Win %:** {cluster_info['avg_win_percentage']:.3f}")
                    st.write(f"**Average Yards/Game:** {cluster_info['avg_yards_per_game']:.1f}")
                    st.write(f"**Average Turnover Margin:** {cluster_info['avg_turnover_margin']:.1f}")
        else:
            st.error(cluster_data['error'])

def render_statistical_comparison(analytics: AdvancedAnalytics, teams: List[str], years: List[int]):
    """Render statistical comparison section."""
    st.header("⚖️ Statistical Comparison")
    st.markdown("Compare two teams using statistical significance tests.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        team1 = st.selectbox("Select Team 1", teams)
    
    with col2:
        team2 = st.selectbox("Select Team 2", teams)
    
    with col3:
        selected_year = st.selectbox("Select Year", years)
    
    metrics = ['win_percentage', 'yards_per_game', 'yards_allowed_per_game', 
              'turnover_margin', 'offensive_efficiency', 'defensive_efficiency']
    selected_metric = st.selectbox("Select Metric", metrics)
    
    if st.button("Compare Teams", type="primary"):
        with st.spinner("Performing statistical comparison..."):
            comparison_data = analytics.calculate_statistical_significance(
                team1, team2, selected_metric, selected_year
            )
        
        if "error" not in comparison_data:
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Comparison Chart")
                fig = AdvancedVisualizations.create_statistical_comparison_chart(comparison_data)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("📋 Statistical Results")
                st.metric(f"{team1} Mean", f"{comparison_data['team1_mean']:.3f}")
                st.metric(f"{team2} Mean", f"{comparison_data['team2_mean']:.3f}")
                st.metric("Difference", f"{comparison_data['difference']:.3f}")
                st.metric("P-Value", f"{comparison_data['p_value']:.4f}")
                st.metric("Effect Size", comparison_data['effect_size'])
                
                # Significance indicator
                if comparison_data['p_value'] < 0.05:
                    st.success(f"✅ {comparison_data['significance']}")
                else:
                    st.warning(f"⚠️ {comparison_data['significance']}")
        else:
            st.error(comparison_data['error'])

def render_predictive_insights(analytics: AdvancedAnalytics, teams: List[str], years: List[int]):
    """Render predictive insights section."""
    st.header("🔮 Predictive Insights")
    st.markdown("Generate predictions for team performance based on historical trends.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_team = st.selectbox("Select Team", teams)
    
    with col2:
        selected_year = st.selectbox("Select Year", years)
    
    if st.button("Generate Predictions", type="primary"):
        with st.spinner("Generating predictions..."):
            predictions_data = analytics.calculate_predictive_insights(selected_team, selected_year)
        
        if "error" not in predictions_data:
            # Display results
            st.subheader("📊 Predictive Analysis")
            fig = AdvancedVisualizations.create_predictive_insights_chart(predictions_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display summary
            st.subheader("📋 Prediction Summary")
            st.info(f"**Overall Trajectory:** {predictions_data['overall_trajectory']}")
            st.info(f"**Confidence Level:** {predictions_data['confidence_level']}")
            st.info(f"**Data Points:** {predictions_data['data_points']}")
            
            # Display individual predictions
            predictions = predictions_data['predictions']
            if predictions:
                st.subheader("📈 Individual Metric Predictions")
                for metric, pred_data in predictions.items():
                    with st.expander(f"{metric.replace('_', ' ').title()}"):
                        st.write(f"**Predicted Value:** {pred_data['predicted_value']:.3f}")
                        st.write(f"**Confidence Interval:** [{pred_data['confidence_interval'][0]:.3f}, {pred_data['confidence_interval'][1]:.3f}]")
                        st.write(f"**Trend:** {pred_data['trend_direction']}")
                        st.write(f"**Trend Strength:** {pred_data['trend_strength']:.3f}")
        else:
            st.error(predictions_data['error'])

def render_conference_analysis(analytics: AdvancedAnalytics, conferences: List[str], years: List[int]):
    """Render conference analysis section."""
    st.header("🏟️ Conference Analysis")
    st.markdown("Analyze conference-wide performance and competitive balance.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_conference = st.selectbox("Select Conference", conferences)
    
    with col2:
        selected_year = st.selectbox("Select Year", years)
    
    if st.button("Analyze Conference", type="primary"):
        with st.spinner("Analyzing conference..."):
            conference_data = analytics.calculate_conference_analysis(selected_conference, selected_year)
        
        if "error" not in conference_data:
            # Display results
            st.subheader("📊 Conference Analysis")
            fig = AdvancedVisualizations.create_conference_analysis_chart(conference_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display summary
            st.subheader("📋 Conference Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Team Count", conference_data['team_count'])
                st.metric("Avg Win %", f"{conference_data['avg_win_percentage']:.3f}")
            
            with col2:
                st.metric("Avg Yards/Game", f"{conference_data['avg_yards_per_game']:.1f}")
                st.metric("Avg Yards Allowed", f"{conference_data['avg_yards_allowed']:.1f}")
            
            with col3:
                st.metric("Avg Turnover Margin", f"{conference_data['avg_turnover_margin']:.1f}")
                st.metric("Competitive Balance", conference_data['competitive_balance'])
            
            # Additional insights
            st.info(f"**Conference Style:** {conference_data['conference_style']}")
            st.info(f"**Top Team:** {conference_data['top_team']}")
            st.info(f"**Bottom Team:** {conference_data['bottom_team']}")
        else:
            st.error(conference_data['error'])

def render_team_insights_summary(analytics: AdvancedAnalytics, teams: List[str], years: List[int]):
    """Render team insights summary section."""
    st.header("📋 Team Insights Summary")
    st.markdown("Get comprehensive insights and recommendations for any team.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_team = st.selectbox("Select Team", teams)
    
    with col2:
        selected_year = st.selectbox("Select Year", years)
    
    if st.button("Generate Insights", type="primary"):
        with st.spinner("Generating insights..."):
            insights_data = analytics.generate_insights_summary(selected_team, selected_year)
        
        if "error" not in insights_data:
            # Display results
            st.subheader("📊 Team Analysis")
            fig = AdvancedVisualizations.create_insights_summary_chart(insights_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display detailed insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("💪 Strengths")
                if insights_data['strengths']:
                    for strength in insights_data['strengths']:
                        st.success(f"✅ {strength}")
                else:
                    st.info("No significant strengths identified")
            
            with col2:
                st.subheader("⚠️ Weaknesses")
                if insights_data['weaknesses']:
                    for weakness in insights_data['weaknesses']:
                        st.error(f"❌ {weakness}")
                else:
                    st.info("No significant weaknesses identified")
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if insights_data['recommendations']:
                for i, recommendation in enumerate(insights_data['recommendations'], 1):
                    st.info(f"{i}. {recommendation}")
            else:
                st.info("No specific recommendations at this time")
            
            # Key metrics
            st.subheader("📈 Key Metrics")
            metrics = insights_data['key_metrics']
            for metric, data in metrics.items():
                st.metric(
                    data['description'],
                    f"{data['value']:.1f}",
                    delta=f"Threshold: {data['threshold']:.1f}"
                )
        else:
            st.error(insights_data['error'])

if __name__ == "__main__":
    main()
