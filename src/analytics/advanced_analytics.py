"""
Advanced Analytics Module for NCAA Football Analytics Platform

This module provides sophisticated analysis capabilities including:
- Trend analysis and forecasting
- Statistical analysis and metrics
- Comparative analytics
- Performance insights
- Predictive modeling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnalytics:
    """Advanced analytics engine for NCAA Football data."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analytics engine.
        
        Args:
            data: DataFrame containing team statistics
        """
        self.data = data.copy()
        self.scaler = StandardScaler()
        
    def calculate_trend_analysis(self, team: str, metric: str, years: List[int] = None) -> Dict:
        """
        Calculate trend analysis for a specific team and metric.
        
        Args:
            team: Team name
            metric: Metric to analyze (e.g., 'win_percentage', 'yards_per_game')
            years: List of years to analyze (default: all available)
            
        Returns:
            Dictionary containing trend analysis results
        """
        if years is None:
            years = sorted(self.data['year'].unique())
            
        team_data = self.data[
            (self.data['team'] == team) & 
            (self.data['year'].isin(years))
        ].sort_values('year')
        
        if len(team_data) < 2:
            return {"error": "Insufficient data for trend analysis"}
            
        values = team_data[metric].values
        years_array = team_data['year'].values
        
        # Calculate trend statistics
        slope, intercept, r_value, p_value, std_err = stats.linregress(years_array, values)
        
        # Calculate year-over-year changes
        yoy_changes = np.diff(values)
        avg_yoy_change = np.mean(yoy_changes)
        
        # Calculate volatility (standard deviation of changes)
        volatility = np.std(yoy_changes)
        
        # Determine trend direction
        if slope > 0.01:
            trend_direction = "Improving"
        elif slope < -0.01:
            trend_direction = "Declining"
        else:
            trend_direction = "Stable"
            
        # Calculate momentum (recent vs historical performance)
        if len(values) >= 3:
            recent_avg = np.mean(values[-2:])  # Last 2 years
            historical_avg = np.mean(values[:-2])  # Earlier years
            momentum = recent_avg - historical_avg
        else:
            momentum = 0
            
        return {
            "team": team,
            "metric": metric,
            "years": years,
            "values": values.tolist(),
            "slope": slope,
            "r_squared": r_value ** 2,
            "p_value": p_value,
            "trend_direction": trend_direction,
            "avg_yoy_change": avg_yoy_change,
            "volatility": volatility,
            "momentum": momentum,
            "current_value": values[-1] if len(values) > 0 else None,
            "best_year": years_array[np.argmax(values)],
            "worst_year": years_array[np.argmin(values)]
        }
    
    def calculate_performance_clusters(self, year: int, metrics: List[str] = None) -> Dict:
        """
        Perform K-means clustering to group teams by performance.
        
        Args:
            year: Year to analyze
            metrics: List of metrics to use for clustering
            
        Returns:
            Dictionary containing clustering results
        """
        if metrics is None:
            metrics = [
                'win_percentage', 'yards_per_game', 'yards_allowed_per_game',
                'turnover_margin', 'offensive_efficiency', 'defensive_efficiency'
            ]
            
        year_data = self.data[self.data['year'] == year].copy()
        
        # Prepare data for clustering
        cluster_data = year_data[metrics].fillna(0)
        
        # Standardize the data
        scaled_data = self.scaler.fit_transform(cluster_data)
        
        # Perform K-means clustering
        n_clusters = 4  # Elite, Good, Average, Poor
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Add cluster labels to data
        year_data['cluster'] = clusters
        
        # Define cluster names based on performance
        cluster_names = {
            0: "Elite",
            1: "Good", 
            2: "Average",
            3: "Poor"
        }
        
        # Calculate cluster statistics
        cluster_stats = {}
        for cluster_id in range(n_clusters):
            cluster_teams = year_data[year_data['cluster'] == cluster_id]
            cluster_stats[cluster_names[cluster_id]] = {
                "teams": cluster_teams['team'].tolist(),
                "count": len(cluster_teams),
                "avg_win_percentage": cluster_teams['win_percentage'].mean(),
                "avg_yards_per_game": cluster_teams['yards_per_game'].mean(),
                "avg_turnover_margin": cluster_teams['turnover_margin'].mean()
            }
            
        return {
            "year": year,
            "clusters": cluster_stats,
            "cluster_centers": kmeans.cluster_centers_.tolist(),
            "metrics_used": metrics
        }
    
    def calculate_statistical_significance(self, team1: str, team2: str, 
                                         metric: str, year: int) -> Dict:
        """
        Calculate statistical significance of differences between two teams.
        
        Args:
            team1: First team name
            team2: Second team name
            metric: Metric to compare
            year: Year to analyze
            
        Returns:
            Dictionary containing statistical test results
        """
        year_data = self.data[self.data['year'] == year]
        
        team1_data = year_data[year_data['team'] == team1][metric].values
        team2_data = year_data[year_data['team'] == team2][metric].values
        
        if len(team1_data) == 0 or len(team2_data) == 0:
            return {"error": "Insufficient data for comparison"}
            
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(team1_data, team2_data)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(team1_data) - 1) * np.var(team1_data, ddof=1) + 
                             (len(team2_data) - 1) * np.var(team2_data, ddof=1)) / 
                            (len(team1_data) + len(team2_data) - 2))
        
        cohens_d = (np.mean(team1_data) - np.mean(team2_data)) / pooled_std
        
        # Determine significance level
        if p_value < 0.001:
            significance = "Highly Significant (p < 0.001)"
        elif p_value < 0.01:
            significance = "Very Significant (p < 0.01)"
        elif p_value < 0.05:
            significance = "Significant (p < 0.05)"
        elif p_value < 0.1:
            significance = "Marginally Significant (p < 0.1)"
        else:
            significance = "Not Significant (p >= 0.1)"
            
        return {
            "team1": team1,
            "team2": team2,
            "metric": metric,
            "year": year,
            "team1_mean": np.mean(team1_data),
            "team2_mean": np.mean(team2_data),
            "difference": np.mean(team1_data) - np.mean(team2_data),
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "significance": significance,
            "effect_size": "Large" if abs(cohens_d) > 0.8 else "Medium" if abs(cohens_d) > 0.5 else "Small"
        }
    
    def calculate_predictive_insights(self, team: str, year: int) -> Dict:
        """
        Calculate predictive insights for a team's future performance.
        
        Args:
            team: Team name
            year: Current year
            
        Returns:
            Dictionary containing predictive insights
        """
        # Get historical data for the team
        team_history = self.data[
            (self.data['team'] == team) & 
            (self.data['year'] < year)
        ].sort_values('year')
        
        if len(team_history) < 2:
            return {"error": "Insufficient historical data"}
            
        # Calculate key metrics trends
        metrics = ['win_percentage', 'yards_per_game', 'turnover_margin', 'offensive_efficiency']
        predictions = {}
        
        for metric in metrics:
            if metric in team_history.columns:
                values = team_history[metric].values
                years = team_history['year'].values
                
                # Linear regression for trend
                slope, intercept, r_value, p_value, std_err = stats.linregress(years, values)
                
                # Predict next year
                next_year_prediction = slope * year + intercept
                
                # Calculate confidence interval
                n = len(values)
                t_value = stats.t.ppf(0.975, n-2)  # 95% confidence
                se_pred = std_err * np.sqrt(1 + 1/n + (year - np.mean(years))**2 / np.sum((years - np.mean(years))**2))
                ci_lower = next_year_prediction - t_value * se_pred
                ci_upper = next_year_prediction + t_value * se_pred
                
                predictions[metric] = {
                    "predicted_value": next_year_prediction,
                    "confidence_interval": [ci_lower, ci_upper],
                    "trend_strength": abs(r_value),
                    "trend_direction": "Improving" if slope > 0 else "Declining"
                }
        
        # Calculate overall performance trajectory
        win_pct_trend = predictions.get('win_percentage', {})
        if win_pct_trend:
            if win_pct_trend['predicted_value'] > 0.7:
                trajectory = "Elite Performance Expected"
            elif win_pct_trend['predicted_value'] > 0.6:
                trajectory = "Strong Performance Expected"
            elif win_pct_trend['predicted_value'] > 0.5:
                trajectory = "Average Performance Expected"
            else:
                trajectory = "Below Average Performance Expected"
        else:
            trajectory = "Insufficient Data"
            
        return {
            "team": team,
            "year": year,
            "predictions": predictions,
            "overall_trajectory": trajectory,
            "data_points": len(team_history),
            "confidence_level": "High" if len(team_history) >= 5 else "Medium" if len(team_history) >= 3 else "Low"
        }
    
    def calculate_conference_analysis(self, conference: str, year: int) -> Dict:
        """
        Perform comprehensive analysis of a conference.
        
        Args:
            conference: Conference name
            year: Year to analyze
            
        Returns:
            Dictionary containing conference analysis
        """
        conf_data = self.data[
            (self.data['conference'] == conference) & 
            (self.data['year'] == year)
        ]
        
        if len(conf_data) == 0:
            return {"error": "No data found for conference"}
            
        # Calculate conference statistics
        conf_stats = {
            "conference": conference,
            "year": year,
            "team_count": len(conf_data),
            "avg_win_percentage": conf_data['win_percentage'].mean(),
            "avg_yards_per_game": conf_data['yards_per_game'].mean(),
            "avg_yards_allowed": conf_data['yards_allowed_per_game'].mean(),
            "avg_turnover_margin": conf_data['turnover_margin'].mean(),
            "conference_strength": conf_data['win_percentage'].std(),  # Lower std = more competitive
            "top_team": conf_data.loc[conf_data['win_percentage'].idxmax(), 'team'],
            "bottom_team": conf_data.loc[conf_data['win_percentage'].idxmin(), 'team']
        }
        
        # Calculate competitive balance
        win_pct_std = conf_data['win_percentage'].std()
        if win_pct_std < 0.1:
            balance = "Highly Competitive"
        elif win_pct_std < 0.15:
            balance = "Competitive"
        elif win_pct_std < 0.2:
            balance = "Moderately Competitive"
        else:
            balance = "Not Very Competitive"
            
        conf_stats["competitive_balance"] = balance
        
        # Calculate offensive vs defensive strength
        offensive_strength = conf_data['yards_per_game'].mean()
        defensive_strength = conf_data['yards_allowed_per_game'].mean()
        
        if offensive_strength > defensive_strength:
            conf_stats["conference_style"] = "Offensive"
        elif defensive_strength > offensive_strength:
            conf_stats["conference_style"] = "Defensive"
        else:
            conf_stats["conference_style"] = "Balanced"
            
        return conf_stats
    
    def generate_insights_summary(self, team: str, year: int) -> Dict:
        """
        Generate a comprehensive insights summary for a team.
        
        Args:
            team: Team name
            year: Year to analyze
            
        Returns:
            Dictionary containing comprehensive insights
        """
        team_data = self.data[
            (self.data['team'] == team) & 
            (self.data['year'] == year)
        ]
        
        if len(team_data) == 0:
            return {"error": "No data found for team"}
            
        team_row = team_data.iloc[0]
        
        # Calculate key insights
        insights = {
            "team": team,
            "year": year,
            "conference": team_row['conference'],
            "win_percentage": team_row['win_percentage'],
            "strengths": [],
            "weaknesses": [],
            "key_metrics": {},
            "recommendations": []
        }
        
        # Identify strengths and weaknesses
        metrics_to_analyze = {
            'win_percentage': (0.6, "Winning Record"),
            'yards_per_game': (400, "Offensive Production"),
            'yards_allowed_per_game': (350, "Defensive Performance"),
            'turnover_margin': (5, "Turnover Control"),
            'offensive_efficiency': (400, "Offensive Efficiency"),
            'defensive_efficiency': (350, "Defensive Efficiency")
        }
        
        for metric, (threshold, description) in metrics_to_analyze.items():
            if metric in team_row and pd.notna(team_row[metric]):
                value = team_row[metric]
                insights["key_metrics"][metric] = {
                    "value": value,
                    "description": description,
                    "threshold": threshold
                }
                
                if value > threshold:
                    insights["strengths"].append(f"{description}: {value:.1f}")
                else:
                    insights["weaknesses"].append(f"{description}: {value:.1f}")
        
        # Generate recommendations
        if len(insights["weaknesses"]) > len(insights["strengths"]):
            insights["recommendations"].append("Focus on improving overall team performance")
        
        if 'yards_per_game' in insights["key_metrics"] and insights["key_metrics"]['yards_per_game']['value'] < 350:
            insights["recommendations"].append("Improve offensive production")
            
        if 'yards_allowed_per_game' in insights["key_metrics"] and insights["key_metrics"]['yards_allowed_per_game']['value'] > 400:
            insights["recommendations"].append("Strengthen defensive performance")
            
        if 'turnover_margin' in insights["key_metrics"] and insights["key_metrics"]['turnover_margin']['value'] < 0:
            insights["recommendations"].append("Focus on turnover control")
        
        return insights
