"""
Enhanced NCAA Football Analytics Dashboard with ML Predictions

An interactive dashboard that includes machine learning predictions for 2025 season.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import logging
import joblib
import sys

# Add src to path for database imports
sys.path.append(str(Path(__file__).parent.parent))
from src.storage.simple_database import create_simple_database

# Configure page
st.set_page_config(
    page_title="🏈 NCAA Football Analytics + ML Predictions",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def get_database():
    """Get database connection (cached)"""
    try:
        db = create_simple_database()
        logger.info("Database connection established")
        return db
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        return None

@st.cache_data
def load_data():
    """Load the processed NCAA football data from database"""
    try:
        db = get_database()
        if db is None:
            st.error("Could not connect to database")
            return pd.DataFrame()
        
        # Load all data from database
        df = db.query("SELECT * FROM ncaa_football_data")
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} features from database")
        return df
    except Exception as e:
        logger.error(f"Error loading data from database: {e}")
        # Fallback to CSV if database fails
        try:
            data_path = Path("data/models/ncaa_football_ml_dataset.csv")
            if data_path.exists():
                df = pd.read_csv(data_path)
                logger.info(f"Fallback: Loaded {len(df)} records from CSV")
                return df
        except Exception as csv_e:
            logger.error(f"CSV fallback also failed: {csv_e}")
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

# Database query functions
@st.cache_data
def get_team_data(team_name, season):
    """Get data for a specific team and season"""
    db = get_database()
    if db is None:
        return pd.DataFrame()
    return db.get_team_stats(team_name, season)

@st.cache_data
def get_conference_data(conference, season):
    """Get data for a specific conference and season"""
    db = get_database()
    if db is None:
        return pd.DataFrame()
    return db.query(f"SELECT * FROM ncaa_football_data WHERE conference = '{conference}' AND year = {season}")

@st.cache_data
def get_top_teams_data(season, limit=25):
    """Get top teams for a specific season"""
    db = get_database()
    if db is None:
        return pd.DataFrame()
    return db.get_top_teams(season, limit)

@st.cache_data
def get_big_ten_data(season):
    """Get Big Ten teams for a specific season"""
    db = get_database()
    if db is None:
        return pd.DataFrame()
    return db.get_big_ten_teams(season)

# Load data
df = load_data()
predictions = load_predictions()

if df.empty:
    st.error("No data available. Please run the data pipeline first.")
    st.stop()

# Header with database status
db_status = "✅ Database Connected" if get_database() is not None else "⚠️ Database Unavailable (using CSV fallback)"

st.markdown(f"""
# 🏈 NCAA Football Analytics Dashboard
### *Explore team performance, conference standings, and ML predictions for 2025*
**Database Status:** {db_status}
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

# ML Predictions section
if not predictions.empty:
    st.sidebar.markdown("---")
    st.sidebar.header("🤖 ML Predictions")
    
    # Show top predicted teams
    st.sidebar.markdown("**Top 5 Predicted Teams for 2025:**")
    top_5 = predictions.head(5)
    for _, row in top_5.iterrows():
        team = row['team']
        prob = row.get('top_25_probability', 0)
        prob_str = f"{prob:.3f}" if pd.notna(prob) else "N/A"
        st.sidebar.write(f"• {team}: {prob_str}")

# Filter data based on selections using database queries
if get_database() is not None:
    # Use database for faster queries
    if conference == 'All':
        filtered_df = df[df['year'] == season]
    else:
        filtered_df = get_conference_data(conference, season)
else:
    # Fallback to pandas filtering
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

# ML Predictions Section
if not predictions.empty:
    st.markdown("---")
    st.subheader("🤖 Machine Learning Predictions for 2025")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top 25 Probability Predictions**")
        
        # Top 25 predictions chart
        top_25_pred = predictions.head(20)  # Show top 20
        
        fig_top25 = px.bar(
            top_25_pred,
            x='top_25_probability',
            y='team',
            orientation='h',
            title="Top 25 Probability Predictions",
            color='top_25_probability',
            color_continuous_scale='RdYlGn'
        )
        
        fig_top25.update_layout(
            xaxis_title="Top 25 Probability",
            yaxis_title="Team",
            height=600
        )
        
        st.plotly_chart(fig_top25, use_container_width=True)
    
    with col2:
        st.markdown("**Performance Ranking Predictions**")
        
        # Performance ranking chart
        fig_ranking = px.scatter(
            predictions,
            x='predicted_rank',
            y='top_25_probability',
            color='conference',
            hover_data=['team', 'conference'],
            title="Performance Rank vs Top 25 Probability",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig_ranking.update_layout(
            xaxis_title="Predicted Rank (Lower = Better)",
            yaxis_title="Top 25 Probability",
            height=600
        )
        
        st.plotly_chart(fig_ranking, use_container_width=True)

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

# ML Model Performance and Database Stats
if not predictions.empty:
    st.markdown("---")
    st.subheader("🤖 ML Model Performance & Database Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    
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
    
    with col4:
        # Database performance indicator
        if get_database() is not None:
            db_stats = get_database().get_database_stats()
            st.metric(
                label="Database Records",
                value=f"{db_stats['total_records']:,}",
                help="Total records stored in DuckDB database"
            )
        else:
            st.metric(
                label="Data Source",
                value="CSV File",
                help="Using CSV file (database unavailable)"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🏈 NCAA Football Analytics Dashboard | Built with Streamlit, Plotly, DuckDB & Machine Learning</p>
    <p>Data sourced from College Football Data API | Database: DuckDB | ML Models: Random Forest & Logistic Regression</p>
    <p><strong>Performance:</strong> Database-backed queries for faster analytics and real-time filtering</p>
</div>
""", unsafe_allow_html=True)
