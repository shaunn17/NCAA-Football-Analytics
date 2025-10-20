"""
NCAA Football Analytics Dashboard - Streamlit Version

A modern, interactive dashboard for exploring NCAA Division I college football data.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import logging

# Configure page
st.set_page_config(
    page_title="🏈 NCAA Football Analytics",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_data
def load_data():
    """Load the processed NCAA football data"""
    try:
        data_path = Path("data/models/ncaa_football_ml_dataset.csv")
        if not data_path.exists():
            st.error(f"Data file not found: {data_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load data
df = load_data()

if df.empty:
    st.error("No data available. Please run the data pipeline first.")
    st.stop()

# Header
st.markdown("""
# 🏈 NCAA Football Analytics Dashboard
### *Explore team performance, conference standings, and predictive analytics*
""")

# Sidebar controls
st.sidebar.header("📊 Dashboard Controls")

# Season filter
seasons = sorted([int(x) for x in df['year'].unique() if not pd.isna(x)])
season = st.sidebar.selectbox(
    "Season:",
    options=seasons,
    index=len(seasons)-1,  # Default to latest season
    format_func=lambda x: f"{x}"
)

# Conference filter
conferences = ['All'] + sorted(df['conference'].dropna().unique().tolist())
conference = st.sidebar.selectbox("Conference:", conferences)

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

# Metric selection
metric_options = {
    'Win Percentage': 'win_percentage',
    'Yards per Game': 'yards_per_game',
    'Yards Allowed per Game': 'yards_allowed_per_game',
    'Turnover Margin': 'turnover_margin',
    'First Down Differential': 'first_down_differential',
    'Offensive Efficiency': 'offensive_efficiency',
    'Defensive Efficiency': 'defensive_efficiency',
    'Conference Rank': 'conference_rank',
    'Conference Dominance': 'conference_dominance'
}

selected_metric = st.sidebar.selectbox(
    "Primary Metric:",
    options=list(metric_options.keys()),
    index=0
)

metric_key = metric_options[selected_metric]

# Filter data based on selections
filtered_df = df[df['year'] == season]
if conference != 'All':
    filtered_df = filtered_df[filtered_df['conference'] == conference]

# Summary metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Teams",
        value=len(filtered_df['team'].unique()),
        delta=None
    )

with col2:
    metric_data = filtered_df[metric_key].dropna()
    avg_value = metric_data.mean() if len(metric_data) > 0 else 0
    st.metric(
        label=f"Avg {selected_metric}",
        value=f"{avg_value:.2f}" if not pd.isna(avg_value) else "N/A"
    )

with col3:
    if len(metric_data) > 0:
        top_team_data = filtered_df[filtered_df[metric_key] == filtered_df[metric_key].max()]
        top_team = top_team_data['team'].iloc[0] if len(top_team_data) > 0 else "N/A"
    else:
        top_team = "N/A"
    st.metric(
        label="Top Performer",
        value=top_team
    )

with col4:
    st.metric(
        label="Conferences",
        value=len(filtered_df['conference'].dropna().unique())
    )

# Main charts
st.markdown("---")

# Row 1: Performance Analysis
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Team Performance Analysis")
    
    # Create scatter plot
    fig_scatter = px.scatter(
        filtered_df,
        x='yards_per_game',
        y='yards_allowed_per_game',
        color=metric_key,
        hover_data=['team', 'conference', 'turnover_margin'],
        title=f"Offensive vs Defensive Performance ({int(season)})",
        color_continuous_scale='RdYlBu_r'
    )
    
    fig_scatter.update_layout(
        xaxis_title="Yards per Game (Offense)",
        yaxis_title="Yards Allowed per Game (Defense)",
        height=500
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    st.subheader("🏆 Conference Comparison")
    
    # Group by conference and calculate average metric
    conference_stats = filtered_df.groupby('conference')[metric_key].mean().reset_index()
    conference_stats = conference_stats.sort_values(metric_key, ascending=False)
    
    fig_bar = px.bar(
        conference_stats,
        x='conference',
        y=metric_key,
        title=f"Conference Comparison - {selected_metric}",
        color=metric_key,
        color_continuous_scale='Viridis'
    )
    
    fig_bar.update_layout(
        xaxis_title="Conference",
        yaxis_title=selected_metric,
        height=500,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)

# Row 2: Team Analysis
if selected_teams:
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Team Profile Radar")
        
        # Create radar chart for selected teams
        team_data = filtered_df[filtered_df['team'].isin(selected_teams)]
        
        if not team_data.empty:
            # Select metrics for radar chart
            radar_metrics = ['win_percentage', 'yards_per_game', 'turnover_margin', 
                           'offensive_efficiency', 'defensive_efficiency', 'first_down_differential']
            
            fig_radar = go.Figure()
            
            for team in selected_teams:
                team_row = team_data[team_data['team'] == team]
                if len(team_row) > 0:
                    values = []
                    for metric in radar_metrics:
                        val = team_row[metric].iloc[0] if not team_row[metric].isna().iloc[0] else 0
                        values.append(val)
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=[m.replace('_', ' ').title() for m in radar_metrics],
                        fill='toself',
                        name=team,
                        line=dict(width=2)
                    ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title=f"Team Performance Profile ({int(season)})",
                height=500
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        st.subheader("📊 Performance Trends")
        
        # Create trend chart
        trend_data = df[df['team'].isin(selected_teams)]
        
        if not trend_data.empty:
            fig_trend = px.line(
                trend_data,
                x='year',
                y=metric_key,
                color='team',
                title=f"Performance Trends - {selected_metric}",
                markers=True
            )
            
            fig_trend.update_layout(
                xaxis_title="Season",
                yaxis_title=selected_metric,
                height=500
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)

# Data table
st.markdown("---")
st.subheader("📋 Detailed Team Data")

# Display data table
display_columns = ['team', 'conference', 'year', 'win_percentage', 'yards_per_game', 
                   'turnover_margin', 'offensive_efficiency', 'defensive_efficiency']

# Format the data for display
display_df = filtered_df[display_columns].copy()
display_df['year'] = display_df['year'].astype(int)
display_df = display_df.round(3)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "win_percentage": st.column_config.NumberColumn(
            "Win %",
            help="Win percentage",
            format="%.3f"
        ),
        "yards_per_game": st.column_config.NumberColumn(
            "Yards/Game",
            help="Yards per game",
            format="%.1f"
        ),
        "turnover_margin": st.column_config.NumberColumn(
            "Turnover Margin",
            help="Turnover margin",
            format="%.1f"
        ),
        "offensive_efficiency": st.column_config.NumberColumn(
            "Offensive Eff",
            help="Offensive efficiency",
            format="%.1f"
        ),
        "defensive_efficiency": st.column_config.NumberColumn(
            "Defensive Eff",
            help="Defensive efficiency",
            format="%.1f"
        )
    }
)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🏈 NCAA Football Analytics Dashboard | Built with Streamlit & Plotly</p>
    <p>Data sourced from College Football Data API</p>
</div>
""", unsafe_allow_html=True)
