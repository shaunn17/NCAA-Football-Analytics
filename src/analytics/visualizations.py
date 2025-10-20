"""
Advanced Visualization Components for NCAA Football Analytics

This module provides sophisticated visualization capabilities for:
- Trend analysis charts
- Statistical comparison plots
- Performance clustering visualizations
- Predictive modeling displays
- Interactive analytics dashboards
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import streamlit as st

class AdvancedVisualizations:
    """Advanced visualization components for analytics."""
    
    @staticmethod
    def create_trend_analysis_chart(trend_data: Dict) -> go.Figure:
        """
        Create a trend analysis chart.
        
        Args:
            trend_data: Dictionary containing trend analysis results
            
        Returns:
            Plotly figure object
        """
        if "error" in trend_data:
            return go.Figure().add_annotation(
                text="Insufficient data for trend analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        fig = go.Figure()
        
        # Add actual data points
        fig.add_trace(go.Scatter(
            x=trend_data['years'],
            y=trend_data['values'],
            mode='markers+lines',
            name='Actual',
            line=dict(color='blue', width=3),
            marker=dict(size=8, color='blue')
        ))
        
        # Add trend line
        years_array = np.array(trend_data['years'])
        trend_line = trend_data['slope'] * years_array + trend_data.get('intercept', 0)
        
        fig.add_trace(go.Scatter(
            x=trend_data['years'],
            y=trend_line,
            mode='lines',
            name=f'Trend ({trend_data["trend_direction"]})',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Add confidence interval if available
        if 'confidence_interval' in trend_data:
            ci_lower = [ci[0] for ci in trend_data['confidence_interval']]
            ci_upper = [ci[1] for ci in trend_data['confidence_interval']]
            
            fig.add_trace(go.Scatter(
                x=trend_data['years'] + trend_data['years'][::-1],
                y=ci_upper + ci_lower[::-1],
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=True
            ))
        
        fig.update_layout(
            title=f"{trend_data['team']} - {trend_data['metric'].replace('_', ' ').title()} Trend Analysis",
            xaxis_title="Year",
            yaxis_title=trend_data['metric'].replace('_', ' ').title(),
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_performance_cluster_chart(cluster_data: Dict) -> go.Figure:
        """
        Create a performance clustering visualization.
        
        Args:
            cluster_data: Dictionary containing clustering results
            
        Returns:
            Plotly figure object
        """
        if "error" in cluster_data:
            return go.Figure().add_annotation(
                text="Insufficient data for clustering",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Create subplot with 2x2 grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Win Percentage Distribution', 'Yards per Game Distribution',
                           'Turnover Margin Distribution', 'Cluster Overview'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Extract data for plotting
        clusters = cluster_data['clusters']
        colors = ['red', 'orange', 'yellow', 'green']
        
        # Plot 1: Win Percentage by Cluster
        cluster_names = list(clusters.keys())
        win_pcts = [clusters[name]['avg_win_percentage'] for name in cluster_names]
        
        fig.add_trace(
            go.Bar(x=cluster_names, y=win_pcts, name='Avg Win %', marker_color=colors),
            row=1, col=1
        )
        
        # Plot 2: Yards per Game by Cluster
        yards = [clusters[name]['avg_yards_per_game'] for name in cluster_names]
        
        fig.add_trace(
            go.Bar(x=cluster_names, y=yards, name='Avg Yards/Game', marker_color=colors),
            row=1, col=2
        )
        
        # Plot 3: Turnover Margin by Cluster
        turnovers = [clusters[name]['avg_turnover_margin'] for name in cluster_names]
        
        fig.add_trace(
            go.Bar(x=cluster_names, y=turnovers, name='Avg Turnover Margin', marker_color=colors),
            row=2, col=1
        )
        
        # Plot 4: Team Count by Cluster
        counts = [clusters[name]['count'] for name in cluster_names]
        
        fig.add_trace(
            go.Pie(labels=cluster_names, values=counts, name='Team Distribution'),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Performance Clustering Analysis - {cluster_data['year']}",
            showlegend=False,
            template='plotly_white',
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_statistical_comparison_chart(comparison_data: Dict) -> go.Figure:
        """
        Create a statistical comparison chart between two teams.
        
        Args:
            comparison_data: Dictionary containing statistical comparison results
            
        Returns:
            Plotly figure object
        """
        if "error" in comparison_data:
            return go.Figure().add_annotation(
                text="Insufficient data for comparison",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Create radar chart for comparison
        metrics = ['win_percentage', 'yards_per_game', 'yards_allowed_per_game', 
                  'turnover_margin', 'offensive_efficiency', 'defensive_efficiency']
        
        # Normalize values for radar chart (0-1 scale)
        team1_values = []
        team2_values = []
        
        for metric in metrics:
            if metric in comparison_data:
                team1_val = comparison_data[f'team1_{metric}'] if f'team1_{metric}' in comparison_data else 0
                team2_val = comparison_data[f'team2_{metric}'] if f'team2_{metric}' in comparison_data else 0
                
                # Normalize to 0-1 scale
                max_val = max(team1_val, team2_val, 1)
                team1_values.append(team1_val / max_val)
                team2_values.append(team2_val / max_val)
            else:
                team1_values.append(0)
                team2_values.append(0)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=team1_values,
            theta=[m.replace('_', ' ').title() for m in metrics],
            fill='toself',
            name=comparison_data['team1'],
            line_color='blue'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=team2_values,
            theta=[m.replace('_', ' ').title() for m in metrics],
            fill='toself',
            name=comparison_data['team2'],
            line_color='red'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            title=f"{comparison_data['team1']} vs {comparison_data['team2']} - {comparison_data['year']}",
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_predictive_insights_chart(predictions_data: Dict) -> go.Figure:
        """
        Create a predictive insights chart.
        
        Args:
            predictions_data: Dictionary containing predictive insights
            
        Returns:
            Plotly figure object
        """
        if "error" in predictions_data:
            return go.Figure().add_annotation(
                text="Insufficient data for predictions",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        predictions = predictions_data['predictions']
        
        if not predictions:
            return go.Figure().add_annotation(
                text="No predictions available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Create subplot for multiple metrics
        metrics = list(predictions.keys())
        n_metrics = len(metrics)
        
        fig = make_subplots(
            rows=n_metrics, cols=1,
            subplot_titles=[f"{metric.replace('_', ' ').title()} Prediction" for metric in metrics],
            vertical_spacing=0.1
        )
        
        for i, metric in enumerate(metrics):
            pred_data = predictions[metric]
            
            # Create confidence interval
            ci_lower = pred_data['confidence_interval'][0]
            ci_upper = pred_data['confidence_interval'][1]
            predicted_value = pred_data['predicted_value']
            
            # Add confidence interval
            fig.add_trace(
                go.Scatter(
                    x=[predicted_value, predicted_value],
                    y=[ci_lower, ci_upper],
                    mode='lines',
                    line=dict(color='rgba(0,100,80,0.2)', width=10),
                    showlegend=False
                ),
                row=i+1, col=1
            )
            
            # Add predicted value
            fig.add_trace(
                go.Scatter(
                    x=[predicted_value],
                    y=[predicted_value],
                    mode='markers',
                    marker=dict(size=15, color='red'),
                    name=f"Predicted {metric}",
                    showlegend=(i == 0)
                ),
                row=i+1, col=1
            )
            
            # Add trend direction
            trend_direction = pred_data['trend_direction']
            fig.add_annotation(
                text=f"Trend: {trend_direction}",
                x=predicted_value,
                y=ci_upper + 0.1,
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red" if trend_direction == "Declining" else "green",
                row=i+1, col=1
            )
        
        fig.update_layout(
            title=f"Predictive Insights for {predictions_data['team']} - {predictions_data['year']}",
            template='plotly_white',
            height=300 * n_metrics
        )
        
        return fig
    
    @staticmethod
    def create_conference_analysis_chart(conference_data: Dict) -> go.Figure:
        """
        Create a conference analysis chart.
        
        Args:
            conference_data: Dictionary containing conference analysis
            
        Returns:
            Plotly figure object
        """
        if "error" in conference_data:
            return go.Figure().add_annotation(
                text="No data found for conference",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Create subplot with multiple metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Win Percentage', 'Average Yards per Game',
                           'Average Yards Allowed', 'Turnover Margin'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Extract metrics
        metrics = {
            'avg_win_percentage': (1, 1, 'Win %'),
            'avg_yards_per_game': (1, 2, 'Yards/Game'),
            'avg_yards_allowed': (2, 1, 'Yards Allowed'),
            'avg_turnover_margin': (2, 2, 'Turnover Margin')
        }
        
        for metric, (row, col, title) in metrics.items():
            if metric in conference_data:
                value = conference_data[metric]
                
                fig.add_trace(
                    go.Bar(x=[title], y=[value], name=title, 
                          marker_color='lightblue'),
                    row=row, col=col
                )
        
        fig.update_layout(
            title=f"{conference_data['conference']} Conference Analysis - {conference_data['year']}",
            showlegend=False,
            template='plotly_white',
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_insights_summary_chart(insights_data: Dict) -> go.Figure:
        """
        Create an insights summary chart.
        
        Args:
            insights_data: Dictionary containing insights summary
            
        Returns:
            Plotly figure object
        """
        if "error" in insights_data:
            return go.Figure().add_annotation(
                text="No data found for team",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Create a comprehensive summary chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Key Metrics', 'Strengths vs Weaknesses',
                           'Performance Breakdown', 'Recommendations'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Plot 1: Key Metrics
        metrics = insights_data['key_metrics']
        metric_names = list(metrics.keys())
        metric_values = [metrics[name]['value'] for name in metric_names]
        
        fig.add_trace(
            go.Bar(x=[name.replace('_', ' ').title() for name in metric_names], 
                  y=metric_values, name='Metrics'),
            row=1, col=1
        )
        
        # Plot 2: Strengths vs Weaknesses
        strengths_count = len(insights_data['strengths'])
        weaknesses_count = len(insights_data['weaknesses'])
        
        fig.add_trace(
            go.Pie(labels=['Strengths', 'Weaknesses'], 
                  values=[strengths_count, weaknesses_count],
                  marker_colors=['green', 'red']),
            row=1, col=2
        )
        
        # Plot 3: Performance Breakdown
        win_pct = insights_data['win_percentage']
        performance_categories = ['Wins', 'Losses']
        performance_values = [win_pct, 1 - win_pct]
        
        fig.add_trace(
            go.Bar(x=performance_categories, y=performance_values, 
                  marker_color=['green', 'red']),
            row=2, col=1
        )
        
        # Plot 4: Recommendations (as text)
        recommendations = insights_data['recommendations']
        if recommendations:
            fig.add_trace(
                go.Bar(x=list(range(len(recommendations))), 
                      y=[1] * len(recommendations),
                      text=recommendations,
                      textposition='auto'),
                row=2, col=2
            )
        
        fig.update_layout(
            title=f"{insights_data['team']} - Comprehensive Analysis - {insights_data['year']}",
            showlegend=False,
            template='plotly_white',
            height=800
        )
        
        return fig
