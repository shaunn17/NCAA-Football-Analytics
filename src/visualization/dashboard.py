"""
NCAA Football Analytics Dashboard

An interactive dashboard for exploring NCAA Division I college football data,
featuring team comparisons, conference analysis, and performance trends.
"""

import dash
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
])

app.title = "🏈 NCAA Football Analytics Dashboard"

# Load data
def load_data():
    """Load the processed NCAA football data"""
    try:
        data_path = Path("data/models/ncaa_football_ml_dataset.csv")
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}")
            return pd.DataFrame()
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load data
df = load_data()

# Dashboard layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🏈 NCAA Football Analytics Dashboard", 
                className="text-center", 
                style={'color': '#1f4e79', 'margin-bottom': '20px'}),
        html.P("Explore team performance, conference standings, and predictive analytics",
               className="text-center text-muted",
               style={'font-size': '18px', 'margin-bottom': '30px'})
    ], className="container-fluid", style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 
                                          'color': 'white', 'padding': '30px 0', 'margin-bottom': '30px'}),
    
    # Main content
    html.Div([
        # Sidebar controls
        html.Div([
            html.H4("📊 Dashboard Controls", style={'color': '#1f4e79', 'margin-bottom': '20px'}),
            
            # Season filter
            html.Label("Season:", style={'font-weight': 'bold', 'margin-top': '10px'}),
            dcc.Dropdown(
                id='season-dropdown',
                options=[{'label': f'{int(year)}', 'value': year} for year in sorted(df['year'].unique())],
                value=df['year'].max(),
                clearable=False,
                style={'margin-bottom': '15px'}
            ),
            
            # Conference filter
            html.Label("Conference:", style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='conference-dropdown',
                options=[{'label': conf, 'value': conf} for conf in sorted(df['conference'].dropna().unique())],
                value='B1G',  # Default to Big Ten
                clearable=True,
                style={'margin-bottom': '15px'}
            ),
            
            # Team filter
            html.Label("Team:", style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='team-dropdown',
                multi=True,
                style={'margin-bottom': '15px'}
            ),
            
            # Metric selection
            html.Label("Primary Metric:", style={'font-weight': 'bold'}),
            dcc.Dropdown(
                id='metric-dropdown',
                options=[
                    {'label': 'Win Percentage', 'value': 'win_percentage'},
                    {'label': 'Yards per Game', 'value': 'yards_per_game'},
                    {'label': 'Yards Allowed per Game', 'value': 'yards_allowed_per_game'},
                    {'label': 'Turnover Margin', 'value': 'turnover_margin'},
                    {'label': 'First Down Differential', 'value': 'first_down_differential'},
                    {'label': 'Offensive Efficiency', 'value': 'offensive_efficiency'},
                    {'label': 'Defensive Efficiency', 'value': 'defensive_efficiency'},
                ],
                value='win_percentage',
                clearable=False,
                style={'margin-bottom': '20px'}
            ),
            
            # Info panel
            html.Div([
                html.H5("ℹ️ Data Info", style={'color': '#1f4e79'}),
                html.P(f"Total Teams: {len(df['team'].unique())}"),
                html.P(f"Seasons: {len(df['year'].unique())}"),
                html.P(f"Conferences: {len(df['conference'].dropna().unique())}"),
                html.P(f"Total Records: {len(df)}")
            ], style={'background': '#f8f9fa', 'padding': '15px', 'border-radius': '5px', 'margin-top': '20px'})
            
        ], className="col-md-3", style={'background': '#f8f9fa', 'padding': '20px', 'border-radius': '10px', 'margin-right': '20px'}),
        
        # Main dashboard content
        html.Div([
            # Key metrics cards
            html.Div([
                html.Div([
                    html.H3(id='total-teams', style={'color': '#28a745', 'margin': '0'}),
                    html.P("Teams Analyzed", style={'margin': '0', 'color': '#6c757d'})
                ], className="card text-center", style={'padding': '20px', 'margin': '5px', 'border-radius': '10px', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                
                html.Div([
                    html.H3(id='avg-metric', style={'color': '#007bff', 'margin': '0'}),
                    html.P(id='avg-metric-label', style={'margin': '0', 'color': '#6c757d'})
                ], className="card text-center", style={'padding': '20px', 'margin': '5px', 'border-radius': '10px', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                
                html.Div([
                    html.H3(id='top-team', style={'color': '#dc3545', 'margin': '0'}),
                    html.P("Top Performer", style={'margin': '0', 'color': '#6c757d'})
                ], className="card text-center", style={'padding': '20px', 'margin': '5px', 'border-radius': '10px', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'}),
                
                html.Div([
                    html.H3(id='conference-count', style={'color': '#ffc107', 'margin': '0'}),
                    html.P("Conferences", style={'margin': '0', 'color': '#6c757d'})
                ], className="card text-center", style={'padding': '20px', 'margin': '5px', 'border-radius': '10px', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], className="row", style={'margin-bottom': '30px'}),
            
            # Charts row
            html.Div([
                # Team performance scatter plot
                html.Div([
                    html.H4("📈 Team Performance Analysis", style={'color': '#1f4e79', 'margin-bottom': '15px'}),
                    dcc.Graph(id='performance-scatter')
                ], className="col-md-6", style={'margin-bottom': '20px'}),
                
                # Conference comparison
                html.Div([
                    html.H4("🏆 Conference Comparison", style={'color': '#1f4e79', 'margin-bottom': '15px'}),
                    dcc.Graph(id='conference-comparison')
                ], className="col-md-6", style={'margin-bottom': '20px'})
            ], className="row"),
            
            # Second row of charts
            html.Div([
                # Team radar chart
                html.Div([
                    html.H4("🎯 Team Profile Radar", style={'color': '#1f4e79', 'margin-bottom': '15px'}),
                    dcc.Graph(id='team-radar')
                ], className="col-md-6", style={'margin-bottom': '20px'}),
                
                # Performance trends
                html.Div([
                    html.H4("📊 Performance Trends", style={'color': '#1f4e79', 'margin-bottom': '15px'}),
                    dcc.Graph(id='trend-analysis')
                ], className="col-md-6", style={'margin-bottom': '20px'})
            ], className="row"),
            
            # Data table
            html.Div([
                html.H4("📋 Detailed Team Data", style={'color': '#1f4e79', 'margin-bottom': '15px'}),
                dash_table.DataTable(
                    id='team-table',
                    columns=[
                        {"name": "Team", "id": "team"},
                        {"name": "Conference", "id": "conference"},
                        {"name": "Season", "id": "year"},
                        {"name": "Win %", "id": "win_percentage", "type": "numeric", "format": {"specifier": ".3f"}},
                        {"name": "Yards/Game", "id": "yards_per_game", "type": "numeric", "format": {"specifier": ".1f"}},
                        {"name": "Turnover Margin", "id": "turnover_margin", "type": "numeric", "format": {"specifier": ".1f"}},
                        {"name": "Offensive Eff", "id": "offensive_efficiency", "type": "numeric", "format": {"specifier": ".1f"}},
                        {"name": "Defensive Eff", "id": "defensive_efficiency", "type": "numeric", "format": {"specifier": ".1f"}}
                    ],
                    style_cell={'textAlign': 'left', 'padding': '10px'},
                    style_header={'backgroundColor': '#1f4e79', 'color': 'white', 'fontWeight': 'bold'},
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 249, 250)'
                        }
                    ],
                    page_size=10,
                    sort_action="native",
                    filter_action="native"
                )
            ], style={'margin-top': '30px'})
            
        ], className="col-md-8")
        
    ], className="row", style={'margin': '0 20px'})
    
], className="container-fluid")

# Callback functions
@app.callback(
    [Output('team-dropdown', 'options'),
     Output('team-dropdown', 'value')],
    [Input('season-dropdown', 'value'),
     Input('conference-dropdown', 'value')]
)
def update_team_dropdown(selected_season, selected_conference):
    """Update team dropdown based on season and conference filters"""
    filtered_df = df[df['year'] == selected_season]
    if selected_conference:
        filtered_df = filtered_df[filtered_df['conference'] == selected_conference]
    
    teams = sorted(filtered_df['team'].unique())
    options = [{'label': team, 'value': team} for team in teams]
    
    # Default to first few teams if Big Ten
    default_value = teams[:3] if selected_conference == 'B1G' and len(teams) >= 3 else teams[:1] if teams else []
    
    return options, default_value

@app.callback(
    [Output('total-teams', 'children'),
     Output('avg-metric', 'children'),
     Output('avg-metric-label', 'children'),
     Output('top-team', 'children'),
     Output('conference-count', 'children')],
    [Input('season-dropdown', 'value'),
     Input('conference-dropdown', 'value'),
     Input('metric-dropdown', 'value')]
)
def update_summary_cards(selected_season, selected_conference, selected_metric):
    """Update summary cards"""
    filtered_df = df[df['year'] == selected_season]
    if selected_conference:
        filtered_df = filtered_df[filtered_df['conference'] == selected_conference]
    
    total_teams = len(filtered_df['team'].unique())
    
    # Calculate average metric
    metric_data = filtered_df[selected_metric].dropna()
    avg_metric = metric_data.mean() if len(metric_data) > 0 else 0
    
    # Find top team
    top_team_data = filtered_df[filtered_df[selected_metric] == filtered_df[selected_metric].max()]
    top_team = top_team_data['team'].iloc[0] if len(top_team_data) > 0 else "N/A"
    
    # Conference count
    conference_count = len(filtered_df['conference'].dropna().unique())
    
    # Format metric label
    metric_labels = {
        'win_percentage': 'Avg Win %',
        'yards_per_game': 'Avg Yards/Game',
        'yards_allowed_per_game': 'Avg Yards Allowed',
        'turnover_margin': 'Avg Turnover Margin',
        'first_down_differential': 'Avg First Down Diff',
        'offensive_efficiency': 'Avg Offensive Eff',
        'defensive_efficiency': 'Avg Defensive Eff'
    }
    
    metric_label = metric_labels.get(selected_metric, 'Avg Metric')
    
    return (total_teams, 
            f"{avg_metric:.2f}" if not pd.isna(avg_metric) else "N/A",
            metric_label,
            top_team,
            conference_count)

@app.callback(
    Output('performance-scatter', 'figure'),
    [Input('season-dropdown', 'value'),
     Input('conference-dropdown', 'value'),
     Input('metric-dropdown', 'value')]
)
def update_performance_scatter(selected_season, selected_conference, selected_metric):
    """Update performance scatter plot"""
    filtered_df = df[df['year'] == selected_season]
    if selected_conference:
        filtered_df = filtered_df[filtered_df['conference'] == selected_conference]
    
    # Create scatter plot
    fig = px.scatter(
        filtered_df, 
        x='yards_per_game', 
        y='yards_allowed_per_game',
        color=selected_metric,
        hover_data=['team', 'conference', 'turnover_margin'],
        title=f"Offensive vs Defensive Performance ({int(selected_season)})",
        color_continuous_scale='RdYlBu_r'
    )
    
    fig.update_layout(
        xaxis_title="Yards per Game (Offense)",
        yaxis_title="Yards Allowed per Game (Defense)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

@app.callback(
    Output('conference-comparison', 'figure'),
    [Input('season-dropdown', 'value'),
     Input('metric-dropdown', 'value')]
)
def update_conference_comparison(selected_season, selected_metric):
    """Update conference comparison chart"""
    filtered_df = df[df['year'] == selected_season]
    
    # Group by conference and calculate average metric
    conference_stats = filtered_df.groupby('conference')[selected_metric].mean().reset_index()
    conference_stats = conference_stats.sort_values(selected_metric, ascending=False)
    
    fig = px.bar(
        conference_stats,
        x='conference',
        y=selected_metric,
        title=f"Conference Comparison - {selected_metric.replace('_', ' ').title()}",
        color=selected_metric,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title="Conference",
        yaxis_title=selected_metric.replace('_', ' ').title(),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_tickangle=-45
    )
    
    return fig

@app.callback(
    Output('team-radar', 'figure'),
    [Input('season-dropdown', 'value'),
     Input('team-dropdown', 'value')]
)
def update_team_radar(selected_season, selected_teams):
    """Update team radar chart"""
    if not selected_teams:
        return go.Figure()
    
    filtered_df = df[(df['year'] == selected_season) & (df['team'].isin(selected_teams))]
    
    # Select metrics for radar chart
    metrics = ['win_percentage', 'yards_per_game', 'turnover_margin', 
               'offensive_efficiency', 'defensive_efficiency', 'first_down_differential']
    
    fig = go.Figure()
    
    for team in selected_teams:
        team_data = filtered_df[filtered_df['team'] == team]
        if len(team_data) > 0:
            values = []
            for metric in metrics:
                val = team_data[metric].iloc[0] if not team_data[metric].isna().iloc[0] else 0
                values.append(val)
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=[m.replace('_', ' ').title() for m in metrics],
                fill='toself',
                name=team,
                line=dict(width=2)
            ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]  # Normalize to 0-1 range
            )),
        showlegend=True,
        title=f"Team Performance Profile ({int(selected_season)})",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

@app.callback(
    Output('trend-analysis', 'figure'),
    [Input('team-dropdown', 'value'),
     Input('metric-dropdown', 'value')]
)
def update_trend_analysis(selected_teams, selected_metric):
    """Update trend analysis chart"""
    if not selected_teams:
        return go.Figure()
    
    filtered_df = df[df['team'].isin(selected_teams)]
    
    fig = px.line(
        filtered_df,
        x='year',
        y=selected_metric,
        color='team',
        title=f"Performance Trends - {selected_metric.replace('_', ' ').title()}",
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Season",
        yaxis_title=selected_metric.replace('_', ' ').title(),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

@app.callback(
    Output('team-table', 'data'),
    [Input('season-dropdown', 'value'),
     Input('conference-dropdown', 'value')]
)
def update_team_table(selected_season, selected_conference):
    """Update team data table"""
    filtered_df = df[df['year'] == selected_season]
    if selected_conference:
        filtered_df = filtered_df[filtered_df['conference'] == selected_conference]
    
    return filtered_df.to_dict('records')

if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8050)
