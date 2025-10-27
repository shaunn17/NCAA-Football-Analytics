#!/usr/bin/env python3
"""
Simple NCAA Football Analytics Dashboard

A simplified version that works without database dependencies.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="🏈 NCAA Football Analytics Dashboard",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load the processed NCAA football data"""
    try:
        data_path = Path("data/models/ncaa_football_ml_dataset.csv")
        if data_path.exists():
            df = pd.read_csv(data_path)
            logger.info(f"Loaded {len(df)} records from CSV")
            return df
        else:
            logger.error(f"Data file not found: {data_path}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_predictions():
    """Load ML predictions"""
    try:
        predictions_path = Path("data/models/2025_predictions.csv")
        if predictions_path.exists():
            predictions = pd.read_csv(predictions_path)
            logger.info(f"Loaded predictions for {len(predictions)} teams")
            return predictions
        else:
            logger.warning("No predictions file found")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading predictions: {e}")
        return pd.DataFrame()

# Load data
df = load_data()
predictions = load_predictions()

if df.empty:
    st.error("🚨 No data available. Please ensure the data pipeline has been run.")
    st.info("Run: python scripts/run_pipeline.py")
    st.stop()

# Header
st.markdown("""
# 🏈 NCAA Football Analytics Dashboard
### *Explore team performance, conference standings, and ML predictions for 2025*
**Data Source:** 📁 CSV Data
""")

# Sidebar controls
st.sidebar.header("📊 Dashboard Controls")

# Season filter
available_seasons = sorted([int(year) for year in df['year'].unique() if pd.notna(year)])
season = st.sidebar.selectbox(
    "Season:",
    options=available_seasons,
    index=len(available_seasons)-1  # Default to latest season
)

# Conference filter
available_conferences = sorted([str(conf) for conf in df[df['year'] == season]['conference'].unique() if pd.notna(conf)])
conference_options = ['All'] + available_conferences
conference = st.sidebar.selectbox(
    "Conference:",
    options=conference_options,
    index=0
)

# Team filter
if conference == 'All':
    available_teams = sorted([str(team) for team in df[df['year'] == season]['team'].unique() if pd.notna(team)])
else:
    available_teams = sorted([str(team) for team in df[(df['year'] == season) & (df['conference'] == conference)]['team'].unique() if pd.notna(team)])

selected_teams = st.sidebar.multiselect(
    "Teams:",
    options=available_teams,
    default=available_teams[:3] if len(available_teams) >= 3 else available_teams
)

# ML Predictions sidebar
if not predictions.empty:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 2025 ML Predictions")
    
    # Top 25 probabilities
    top_25_predictions = predictions.nlargest(10, 'top_25_probability')
    st.sidebar.write("**Top 25 Probabilities:**")
    for _, row in top_25_predictions.iterrows():
        team = row['team']
        prob = row.get('top_25_probability', 0)
        prob_str = f"{prob:.3f}" if pd.notna(prob) else "N/A"
        st.sidebar.write(f"• {team}: {prob_str}")

# Filter data based on selections
filtered_df = df[df['year'] == season]
if conference != 'All':
    filtered_df = filtered_df[filtered_df['conference'] == conference]

# Summary metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_teams = len(filtered_df)
    st.metric(
        label="Total Teams",
        value=total_teams,
        help=f"Number of teams in {conference if conference != 'All' else 'all conferences'} for {season}"
    )

with col2:
    avg_win_pct = filtered_df['win_percentage'].mean()
    st.metric(
        label="Avg Win %",
        value=f"{avg_win_pct:.3f}" if pd.notna(avg_win_pct) else "N/A",
        help=f"Average win percentage for {conference if conference != 'All' else 'all conferences'} in {season}"
    )

with col3:
    avg_yards = filtered_df['yards_per_game'].mean()
    st.metric(
        label="Avg Yards/Game",
        value=f"{avg_yards:.1f}" if pd.notna(avg_yards) else "N/A",
        help=f"Average yards per game for {conference if conference != 'All' else 'all conferences'} in {season}"
    )

with col4:
    if not predictions.empty:
        pred_count = len(predictions)
        st.metric(
            label="2025 Predictions",
            value=pred_count,
            help="Number of teams with ML predictions for 2025"
        )
    else:
        st.metric(
            label="Data Records",
            value=len(df),
            help="Total records in dataset"
        )

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Team Performance", "🏟️ Conference Analysis", "🤖 ML Predictions", "📈 Advanced Analytics"])

with tab1:
    st.subheader(f"Team Performance Analysis - {season}")
    
    if selected_teams:
        # Team comparison chart
        team_data = filtered_df[filtered_df['team'].isin(selected_teams)]
        
        if not team_data.empty:
            # Win percentage comparison
            fig_win = px.bar(
                team_data, 
                x='team', 
                y='win_percentage',
                title=f"Win Percentage Comparison - {season}",
                color='win_percentage',
                color_continuous_scale='RdYlGn'
            )
            fig_win.update_layout(xaxis_title="Team", yaxis_title="Win Percentage")
            st.plotly_chart(fig_win, use_container_width=True)
            
            # Yards per game comparison
            fig_yards = px.bar(
                team_data, 
                x='team', 
                y='yards_per_game',
                title=f"Yards Per Game - {season}",
                color='yards_per_game',
                color_continuous_scale='Blues'
            )
            fig_yards.update_layout(xaxis_title="Team", yaxis_title="Yards Per Game")
            st.plotly_chart(fig_yards, use_container_width=True)
            
            # Team stats table
            st.subheader("Team Statistics")
            display_cols = ['team', 'win_percentage', 'yards_per_game', 'yards_allowed_per_game', 'turnover_margin']
            available_cols = [col for col in display_cols if col in team_data.columns]
            st.dataframe(
                team_data[available_cols].round(3),
                use_container_width=True
            )
    else:
        st.info("👆 Select teams from the sidebar to see detailed comparisons")

with tab2:
    st.subheader(f"Conference Analysis - {season}")
    
    # Conference standings
    if conference == 'All':
        conf_data = filtered_df.groupby('conference').agg({
            'win_percentage': 'mean',
            'yards_per_game': 'mean',
            'team': 'count'
        }).round(3)
        conf_data.columns = ['Avg Win %', 'Avg Yards/Game', 'Team Count']
        conf_data = conf_data.sort_values('Avg Win %', ascending=False)
        
        st.subheader("Conference Rankings")
        st.dataframe(conf_data, use_container_width=True)
        
        # Conference comparison chart
        fig_conf = px.bar(
            conf_data.reset_index(),
            x='conference',
            y='Avg Win %',
            title=f"Conference Win Percentage Comparison - {season}",
            color='Avg Win %',
            color_continuous_scale='RdYlGn'
        )
        fig_conf.update_layout(xaxis_title="Conference", yaxis_title="Average Win Percentage")
        st.plotly_chart(fig_conf, use_container_width=True)
    else:
        # Individual conference standings
        conf_teams = filtered_df.sort_values('win_percentage', ascending=False)
        
        st.subheader(f"{conference} Conference Standings")
        display_cols = ['team', 'win_percentage', 'yards_per_game', 'yards_allowed_per_game', 'turnover_margin']
        available_cols = [col for col in display_cols if col in conf_teams.columns]
        st.dataframe(
            conf_teams[available_cols].round(3),
            use_container_width=True
        )

with tab3:
    st.subheader("🤖 Machine Learning Predictions for 2025")
    
    if not predictions.empty:
        # Top 25 predictions
        st.subheader("Top 25 Team Predictions")
        top_25 = predictions.nlargest(25, 'top_25_probability')
        
        fig_top25 = px.bar(
            top_25,
            x='team',
            y='top_25_probability',
            title="Top 25 Team Probabilities for 2025",
            color='top_25_probability',
            color_continuous_scale='RdYlGn'
        )
        fig_top25.update_layout(xaxis_title="Team", yaxis_title="Top 25 Probability")
        st.plotly_chart(fig_top25, use_container_width=True)
        
        # Performance rankings
        st.subheader("Performance Rankings")
        perf_rankings = predictions.sort_values('predicted_rank', ascending=True)
        
        fig_perf = px.scatter(
            perf_rankings.head(50),
            x='predicted_rank',
            y='top_25_probability',
            hover_data=['team'],
            title="Performance Ranking vs Top 25 Probability",
            color='top_25_probability',
            color_continuous_scale='RdYlGn'
        )
        fig_perf.update_layout(xaxis_title="Predicted Rank", yaxis_title="Top 25 Probability")
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # Predictions table
        st.subheader("All Predictions")
        pred_cols = ['team', 'top_25_probability', 'predicted_rank']
        available_pred_cols = [col for col in pred_cols if col in predictions.columns]
        st.dataframe(
            predictions[available_pred_cols].round(3),
            use_container_width=True
        )
    else:
        st.warning("⚠️ No ML predictions available. Please run the ML training pipeline.")

with tab4:
    st.subheader("📈 Advanced Analytics")
    
    # Correlation analysis
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
    correlation_cols = ['win_percentage', 'yards_per_game', 'yards_allowed_per_game', 'turnover_margin']
    available_corr_cols = [col for col in correlation_cols if col in numeric_cols]
    
    if len(available_corr_cols) > 1:
        corr_matrix = filtered_df[available_corr_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Performance Metrics Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Scatter plot analysis
    if 'win_percentage' in filtered_df.columns and 'yards_per_game' in filtered_df.columns:
        fig_scatter = px.scatter(
            filtered_df,
            x='yards_per_game',
            y='win_percentage',
            color='conference',
            hover_data=['team'],
            title="Yards Per Game vs Win Percentage",
            labels={'yards_per_game': 'Yards Per Game', 'win_percentage': 'Win Percentage'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ML Model Performance
if not predictions.empty:
    st.markdown("---")
    st.subheader("🤖 ML Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Top 25 Model Accuracy",
            value="81.5%",
            help="Accuracy of the Random Forest model for predicting top 25 teams"
        )
    
    with col2:
        st.metric(
            label="Ranking Model R²",
            value="97.7%",
            help="R² score of the Random Forest model for performance ranking"
        )
    
    with col3:
        st.metric(
            label="Predictions Generated",
            value=f"{len(predictions)}",
            help="Number of teams with 2025 season predictions"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🏈 NCAA Football Analytics Dashboard | Built with Streamlit, Plotly & Machine Learning</p>
    <p>Data sourced from College Football Data API | ML Models: Random Forest & Logistic Regression</p>
</div>
""", unsafe_allow_html=True)
