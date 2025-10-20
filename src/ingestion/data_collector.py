"""
Data Collector for NCAA Football Analytics Platform

This module orchestrates the collection of data from the College Football Data API,
handling multiple seasons, teams, and data types with proper error handling and logging.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from .api_client import create_api_client
from config.settings import settings


logger = logging.getLogger(__name__)


class DataCollector:
    """Main data collection orchestrator"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the data collector
        
        Args:
            api_key: College Football Data API key
        """
        self.api_client = create_api_client()
        self.raw_data_dir = settings.raw_data_dir
        self.seasons = settings.seasons_to_collect
        self.conferences = settings.conferences_to_focus
        
        # Ensure raw data directory exists
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Data collector initialized for seasons: {self.seasons}")
        logger.info(f"Focus conferences: {self.conferences}")
    
    def collect_all_data(self, seasons: List[int] = None, 
                        conferences: List[str] = None) -> Dict[str, Any]:
        """
        Collect all data types for specified seasons and conferences
        
        Args:
            seasons: List of seasons to collect (defaults to settings)
            conferences: List of conferences to focus on (defaults to settings)
            
        Returns:
            Dictionary with collection results and statistics
        """
        seasons = seasons or self.seasons
        conferences = conferences or self.conferences
        
        collection_results = {
            "start_time": datetime.now().isoformat(),
            "seasons": seasons,
            "conferences": conferences,
            "results": {}
        }
        
        try:
            # Test API connection first
            if not self.api_client.test_connection():
                raise Exception("API connection test failed")
            
            # Collect basic reference data
            logger.info("Collecting reference data...")
            collection_results["results"]["conferences"] = self.collect_conferences()
            collection_results["results"]["teams"] = self.collect_teams()
            
            # Collect season-specific data
            for season in seasons:
                logger.info(f"Collecting data for season {season}...")
                season_results = self.collect_season_data(season, conferences)
                collection_results["results"][f"season_{season}"] = season_results
            
            collection_results["end_time"] = datetime.now().isoformat()
            collection_results["success"] = True
            
            # Save collection metadata
            self._save_collection_metadata(collection_results)
            
            logger.info("Data collection completed successfully")
            return collection_results
            
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            collection_results["error"] = str(e)
            collection_results["success"] = False
            collection_results["end_time"] = datetime.now().isoformat()
            return collection_results
    
    def collect_conferences(self) -> Dict[str, Any]:
        """Collect conference information"""
        try:
            conferences_data = self.api_client.get_conferences()
            
            # Save raw data
            conferences_file = self.raw_data_dir / "conferences.json"
            with open(conferences_file, 'w') as f:
                json.dump(conferences_data, f, indent=2)
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(conferences_data)
            csv_file = self.raw_data_dir / "conferences.csv"
            df.to_csv(csv_file, index=False)
            
            logger.info(f"Collected {len(conferences_data)} conferences")
            return {
                "count": len(conferences_data),
                "file": str(conferences_file),
                "csv_file": str(csv_file)
            }
            
        except Exception as e:
            logger.error(f"Failed to collect conferences: {e}")
            raise
    
    def collect_teams(self, conference: str = None) -> Dict[str, Any]:
        """Collect team information"""
        try:
            teams_data = self.api_client.get_teams(conference)
            
            # Save raw data
            teams_file = self.raw_data_dir / f"teams_{conference or 'all'}.json"
            with open(teams_file, 'w') as f:
                json.dump(teams_data, f, indent=2)
            
            # Convert to DataFrame
            df = pd.DataFrame(teams_data)
            csv_file = self.raw_data_dir / f"teams_{conference or 'all'}.csv"
            df.to_csv(csv_file, index=False)
            
            logger.info(f"Collected {len(teams_data)} teams for {conference or 'all conferences'}")
            return {
                "count": len(teams_data),
                "file": str(teams_file),
                "csv_file": str(csv_file),
                "conference": conference
            }
            
        except Exception as e:
            logger.error(f"Failed to collect teams for {conference}: {e}")
            raise
    
    def collect_season_data(self, season: int, conferences: List[str] = None) -> Dict[str, Any]:
        """
        Collect all data for a specific season
        
        Args:
            season: Season year
            conferences: List of conferences to focus on
            
        Returns:
            Dictionary with collection results for the season
        """
        season_results = {
            "season": season,
            "start_time": datetime.now().isoformat(),
            "data_types": {}
        }
        
        try:
            # Collect team statistics
            logger.info(f"Collecting team stats for season {season}")
            season_results["data_types"]["team_stats"] = self.collect_team_stats(season)
            
            # Collect games
            logger.info(f"Collecting games for season {season}")
            season_results["data_types"]["games"] = self.collect_games(season)
            
            # Collect rankings
            logger.info(f"Collecting rankings for season {season}")
            season_results["data_types"]["rankings"] = self.collect_rankings(season)
            
            # Collect drives (optional - can be large)
            logger.info(f"Collecting drives for season {season}")
            season_results["data_types"]["drives"] = self.collect_drives(season)
            
            season_results["end_time"] = datetime.now().isoformat()
            season_results["success"] = True
            
            logger.info(f"Season {season} data collection completed")
            return season_results
            
        except Exception as e:
            logger.error(f"Failed to collect data for season {season}: {e}")
            season_results["error"] = str(e)
            season_results["success"] = False
            season_results["end_time"] = datetime.now().isoformat()
            return season_results
    
    def collect_team_stats(self, season: int) -> Dict[str, Any]:
        """Collect team statistics for a season"""
        try:
            team_stats = self.api_client.get_team_stats(season)
            
            # Save raw data
            stats_file = self.raw_data_dir / f"team_stats_{season}.json"
            with open(stats_file, 'w') as f:
                json.dump(team_stats, f, indent=2)
            
            # Convert to DataFrame
            df = pd.DataFrame(team_stats)
            csv_file = self.raw_data_dir / f"team_stats_{season}.csv"
            df.to_csv(csv_file, index=False)
            
            logger.info(f"Collected team stats for {len(team_stats)} teams in {season}")
            return {
                "count": len(team_stats),
                "file": str(stats_file),
                "csv_file": str(csv_file)
            }
            
        except Exception as e:
            logger.error(f"Failed to collect team stats for {season}: {e}")
            raise
    
    def collect_games(self, season: int, weeks: List[int] = None) -> Dict[str, Any]:
        """Collect games for a season"""
        try:
            all_games = []
            
            if weeks:
                # Collect specific weeks
                for week in weeks:
                    games = self.api_client.get_games(season, week=week)
                    all_games.extend(games)
            else:
                # Collect all games for the season
                games = self.api_client.get_games(season)
                all_games.extend(games)
            
            # Save raw data
            games_file = self.raw_data_dir / f"games_{season}.json"
            with open(games_file, 'w') as f:
                json.dump(all_games, f, indent=2)
            
            # Convert to DataFrame
            df = pd.DataFrame(all_games)
            csv_file = self.raw_data_dir / f"games_{season}.csv"
            df.to_csv(csv_file, index=False)
            
            logger.info(f"Collected {len(all_games)} games for {season}")
            return {
                "count": len(all_games),
                "file": str(games_file),
                "csv_file": str(csv_file),
                "weeks": weeks
            }
            
        except Exception as e:
            logger.error(f"Failed to collect games for {season}: {e}")
            raise
    
    def collect_rankings(self, season: int, weeks: List[int] = None) -> Dict[str, Any]:
        """Collect rankings for a season"""
        try:
            all_rankings = []
            
            if weeks:
                # Collect specific weeks
                for week in weeks:
                    rankings = self.api_client.get_rankings(season, week=week)
                    all_rankings.extend(rankings)
            else:
                # Collect all rankings for the season
                rankings = self.api_client.get_rankings(season)
                all_rankings.extend(rankings)
            
            # Save raw data
            rankings_file = self.raw_data_dir / f"rankings_{season}.json"
            with open(rankings_file, 'w') as f:
                json.dump(all_rankings, f, indent=2)
            
            # Convert to DataFrame
            df = pd.DataFrame(all_rankings)
            csv_file = self.raw_data_dir / f"rankings_{season}.csv"
            df.to_csv(csv_file, index=False)
            
            logger.info(f"Collected {len(all_rankings)} ranking entries for {season}")
            return {
                "count": len(all_rankings),
                "file": str(rankings_file),
                "csv_file": str(csv_file),
                "weeks": weeks
            }
            
        except Exception as e:
            logger.error(f"Failed to collect rankings for {season}: {e}")
            raise
    
    def collect_drives(self, season: int, weeks: List[int] = None) -> Dict[str, Any]:
        """Collect drive data for a season (optional - can be large)"""
        try:
            all_drives = []
            
            if weeks:
                # Collect specific weeks
                for week in weeks:
                    drives = self.api_client.get_drives(season, week=week)
                    all_drives.extend(drives)
            else:
                # Collect all drives for the season (this might be very large)
                drives = self.api_client.get_drives(season)
                all_drives.extend(drives)
            
            # Save raw data
            drives_file = self.raw_data_dir / f"drives_{season}.json"
            with open(drives_file, 'w') as f:
                json.dump(all_drives, f, indent=2)
            
            # Convert to DataFrame
            df = pd.DataFrame(all_drives)
            csv_file = self.raw_data_dir / f"drives_{season}.csv"
            df.to_csv(csv_file, index=False)
            
            logger.info(f"Collected {len(all_drives)} drives for {season}")
            return {
                "count": len(all_drives),
                "file": str(drives_file),
                "csv_file": str(csv_file),
                "weeks": weeks
            }
            
        except Exception as e:
            logger.error(f"Failed to collect drives for {season}: {e}")
            raise
    
    def _save_collection_metadata(self, metadata: Dict[str, Any]):
        """Save collection metadata for tracking"""
        metadata_file = self.raw_data_dir / "collection_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Collection metadata saved to {metadata_file}")


def main():
    """Main function for running data collection"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect NCAA Football Data")
    parser.add_argument("--seasons", nargs="+", type=int, 
                       help="Seasons to collect (default: from settings)")
    parser.add_argument("--conferences", nargs="+", 
                       help="Conferences to focus on (default: from settings)")
    parser.add_argument("--api-key", help="College Football Data API key")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create collector and run
    collector = DataCollector(args.api_key)
    results = collector.collect_all_data(args.seasons, args.conferences)
    
    print(f"Collection completed: {results['success']}")
    if not results['success']:
        print(f"Error: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()


