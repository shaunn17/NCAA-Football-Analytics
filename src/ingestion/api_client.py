"""
College Football Data API Client

This module provides a client for interacting with the College Football Data API
to fetch team statistics, game data, and other football-related information.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import settings


logger = logging.getLogger(__name__)


class CollegeFootballDataAPIClient:
    """Client for College Football Data API with rate limiting and error handling"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the API client
        
        Args:
            api_key: College Football Data API key. If None, uses settings value.
        """
        self.api_key = api_key or settings.college_football_data_api_key
        self.base_url = settings.api_base_url
        self.timeout = settings.api_timeout
        self.rate_limit = settings.api_rate_limit
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Rate limiting
        self.last_request_time = 0
        self.request_count = 0
        self.rate_limit_window_start = time.time()
        
        if not self.api_key:
            logger.warning("No API key provided. Some endpoints may not work.")
    
    def _handle_rate_limiting(self):
        """Handle API rate limiting"""
        current_time = time.time()
        
        # Reset counter every minute
        if current_time - self.rate_limit_window_start >= 60:
            self.request_count = 0
            self.rate_limit_window_start = current_time
        
        # Check if we've hit the rate limit
        if self.request_count >= self.rate_limit:
            sleep_time = 60 - (current_time - self.rate_limit_window_start)
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
                self.request_count = 0
                self.rate_limit_window_start = time.time()
        
        # Ensure minimum time between requests
        time_since_last_request = current_time - self.last_request_time
        min_interval = 60 / self.rate_limit  # Minimum seconds between requests
        if time_since_last_request < min_interval:
            sleep_time = min_interval - time_since_last_request
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict[str, Any]:
        """
        Make a request to the API with error handling
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            
        Returns:
            JSON response data
            
        Raises:
            requests.RequestException: If API request fails
        """
        self._handle_rate_limiting()
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        
        try:
            logger.debug(f"Making request to {url} with params: {params}")
            response = self.session.get(
                url, 
                params=params, 
                headers=headers, 
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def get_teams(self, conference: str = None) -> List[Dict[str, Any]]:
        """
        Get list of teams
        
        Args:
            conference: Optional conference filter
            
        Returns:
            List of team information
        """
        params = {}
        if conference:
            params["conference"] = conference
            
        return self._make_request("teams", params)
    
    def get_team_stats(self, year: int, team: str = None) -> List[Dict[str, Any]]:
        """
        Get team statistics for a given year
        
        Args:
            year: Season year
            team: Optional team name filter
            
        Returns:
            List of team statistics
        """
        params = {"year": year}
        if team:
            params["team"] = team
            
        return self._make_request("stats/season", params)
    
    def get_games(self, year: int, week: int = None, season_type: str = None, 
                  team: str = None) -> List[Dict[str, Any]]:
        """
        Get games for a given year
        
        Args:
            year: Season year
            week: Optional week filter
            season_type: Optional season type (regular, postseason, etc.)
            team: Optional team filter
            
        Returns:
            List of game information
        """
        params = {"year": year}
        if week is not None:
            params["week"] = week
        if season_type:
            params["seasonType"] = season_type
        if team:
            params["team"] = team
            
        return self._make_request("games", params)
    
    def get_conferences(self) -> List[Dict[str, Any]]:
        """
        Get list of conferences
        
        Returns:
            List of conference information
        """
        return self._make_request("conferences")
    
    def get_rankings(self, year: int, week: int = None, 
                     season_type: str = None) -> List[Dict[str, Any]]:
        """
        Get rankings for a given year
        
        Args:
            year: Season year
            week: Optional week filter
            season_type: Optional season type
            
        Returns:
            List of ranking information
        """
        params = {"year": year}
        if week is not None:
            params["week"] = week
        if season_type:
            params["seasonType"] = season_type
            
        return self._make_request("rankings", params)
    
    def get_drives(self, year: int, week: int = None, season_type: str = None,
                   team: str = None) -> List[Dict[str, Any]]:
        """
        Get drive data for a given year
        
        Args:
            year: Season year
            week: Optional week filter
            season_type: Optional season type
            team: Optional team filter
            
        Returns:
            List of drive information
        """
        params = {"year": year}
        if week is not None:
            params["week"] = week
        if season_type:
            params["seasonType"] = season_type
        if team:
            params["team"] = team
            
        return self._make_request("drives", params)
    
    def get_plays(self, year: int, week: int = None, season_type: str = None,
                  team: str = None, offense: str = None, defense: str = None) -> List[Dict[str, Any]]:
        """
        Get play-by-play data for a given year
        
        Args:
            year: Season year
            week: Optional week filter
            season_type: Optional season type
            team: Optional team filter
            offense: Optional offensive team filter
            defense: Optional defensive team filter
            
        Returns:
            List of play information
        """
        params = {"year": year}
        if week is not None:
            params["week"] = week
        if season_type:
            params["seasonType"] = season_type
        if team:
            params["team"] = team
        if offense:
            params["offense"] = offense
        if defense:
            params["defense"] = defense
            
        return self._make_request("plays", params)
    
    def test_connection(self) -> bool:
        """
        Test API connection
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.get_conferences()
            logger.info("API connection test successful")
            return True
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return False


def create_api_client() -> CollegeFootballDataAPIClient:
    """Factory function to create an API client with settings"""
    return CollegeFootballDataAPIClient(settings.college_football_data_api_key)


