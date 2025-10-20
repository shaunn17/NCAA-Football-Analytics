#!/usr/bin/env python3
"""
Win Percentage Calculator

This script calculates win percentages from games data and updates the processed dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_win_percentages_from_games():
    """Calculate win percentages from games data"""
    logger.info("Calculating win percentages from games data...")
    
    # Load games data for both years
    games_2023 = pd.read_csv("data/raw/games_2023.csv")
    games_2024 = pd.read_csv("data/raw/games_2024.csv")
    
    # Combine games data
    all_games = pd.concat([games_2023, games_2024], ignore_index=True)
    
    # Filter for completed games only
    completed_games = all_games[all_games['completed'] == True].copy()
    
    logger.info(f"Processing {len(completed_games)} completed games")
    
    # Calculate wins and losses for each team
    team_records = {}
    
    for _, game in completed_games.iterrows():
        home_team = game['homeTeam']
        away_team = game['awayTeam']
        home_points = game['homePoints']
        away_points = game['awayPoints']
        season = int(game['season'])
        
        # Skip games with missing data
        if pd.isna(home_points) or pd.isna(away_points):
            continue
            
        # Initialize team records if not exists
        if home_team not in team_records:
            team_records[home_team] = {}
        if away_team not in team_records:
            team_records[away_team] = {}
            
        if season not in team_records[home_team]:
            team_records[home_team][season] = {'wins': 0, 'losses': 0, 'games': 0}
        if season not in team_records[away_team]:
            team_records[away_team][season] = {'wins': 0, 'losses': 0, 'games': 0}
        
        # Count games
        team_records[home_team][season]['games'] += 1
        team_records[away_team][season]['games'] += 1
        
        # Determine winner
        if home_points > away_points:
            team_records[home_team][season]['wins'] += 1
            team_records[away_team][season]['losses'] += 1
        elif away_points > home_points:
            team_records[away_team][season]['wins'] += 1
            team_records[home_team][season]['losses'] += 1
        # Ties are counted as 0.5 wins for both teams
        else:
            team_records[home_team][season]['wins'] += 0.5
            team_records[away_team][season]['wins'] += 0.5
    
    # Convert to DataFrame
    records_data = []
    for team, seasons in team_records.items():
        for season, record in seasons.items():
            records_data.append({
                'team': team,
                'year': season,
                'wins': record['wins'],
                'losses': record['losses'],
                'games': record['games'],
                'win_percentage': record['wins'] / record['games'] if record['games'] > 0 else 0
            })
    
    records_df = pd.DataFrame(records_data)
    logger.info(f"Calculated records for {len(records_df)} team-season combinations")
    
    return records_df

def update_processed_data_with_win_percentages():
    """Update the processed data with calculated win percentages"""
    logger.info("Updating processed data with win percentages...")
    
    # Calculate win percentages
    win_records = calculate_win_percentages_from_games()
    
    # Load the current processed data
    processed_df = pd.read_csv("data/models/ncaa_football_ml_dataset.csv")
    
    # Merge win percentages into processed data
    # Create a mapping for team names (handle potential name differences)
    team_mapping = {}
    for team in win_records['team'].unique():
        # Find matching team in processed data
        matching_teams = processed_df[processed_df['team'].str.contains(team, case=False, na=False)]['team'].unique()
        if len(matching_teams) > 0:
            team_mapping[team] = matching_teams[0]
    
    # Update win percentages
    updated_count = 0
    for _, record in win_records.iterrows():
        team = record['team']
        year = int(record['year'])
        
        # Find matching team in processed data
        matching_team = team_mapping.get(team, team)
        
        # Update the win percentage
        mask = (processed_df['team'] == matching_team) & (processed_df['year'] == year)
        if mask.any():
            processed_df.loc[mask, 'win_percentage'] = record['win_percentage']
            processed_df.loc[mask, 'wins'] = record['wins']
            processed_df.loc[mask, 'losses'] = record['losses']
            updated_count += 1
    
    logger.info(f"Updated win percentages for {updated_count} team-season combinations")
    
    # Save updated data
    output_path = "data/models/ncaa_football_ml_dataset.csv"
    processed_df.to_csv(output_path, index=False)
    logger.info(f"Updated data saved to {output_path}")
    
    # Show sample of updated data
    print("\n📊 Sample of Updated Data:")
    print("-" * 40)
    sample_data = processed_df[['team', 'year', 'win_percentage', 'wins', 'losses', 'games']].head(10)
    print(sample_data.to_string(index=False))
    
    return processed_df

def main():
    """Main function"""
    print("🏈 NCAA Football Analytics - Win Percentage Calculator")
    print("=" * 60)
    
    try:
        updated_df = update_processed_data_with_win_percentages()
        
        # Show statistics
        print(f"\n📈 Statistics:")
        print(f"   Total records: {len(updated_df)}")
        print(f"   Records with win percentage: {updated_df['win_percentage'].notna().sum()}")
        print(f"   Average win percentage: {updated_df['win_percentage'].mean():.3f}")
        print(f"   Max win percentage: {updated_df['win_percentage'].max():.3f}")
        print(f"   Min win percentage: {updated_df['win_percentage'].min():.3f}")
        
        print(f"\n✅ Win percentage calculation complete!")
        
    except Exception as e:
        logger.error(f"Error calculating win percentages: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
