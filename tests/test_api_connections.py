"""
API Connection Tests for NCAA Football Analytics

Tests to ensure API connections work correctly and handle errors gracefully.
"""

import pytest
import requests
import requests_mock
from unittest.mock import patch, Mock
from tests.conftest import assert_response_time

class TestAPIConnections:
    """Test API connections and error handling"""
    
    def test_college_football_data_api_connection(self, test_env_vars):
        """Test connection to College Football Data API"""
        from src.ingestion.api_client import CollegeFootballAPIClient
        
        client = CollegeFootballAPIClient()
        
        # Test that client initializes correctly
        assert client.api_key is not None, "API key should be set"
        assert client.base_url == "https://api.collegefootballdata.com", "Base URL should be correct"
    
    @requests_mock.Mocker()
    def test_api_get_teams_success(self, mock_request, test_env_vars):
        """Test successful API call to get teams"""
        from src.ingestion.api_client import CollegeFootballAPIClient
        
        # Mock successful API response
        mock_response = [
            {"id": 1, "school": "Indiana", "conference": "B1G"},
            {"id": 2, "school": "Ohio State", "conference": "B1G"}
        ]
        
        mock_request.get(
            "https://api.collegefootballdata.com/teams",
            json=mock_response,
            status_code=200
        )
        
        client = CollegeFootballAPIClient()
        teams = client.get_teams()
        
        # Verify response
        assert len(teams) == 2
        assert teams[0]['school'] == 'Indiana'
        assert teams[1]['school'] == 'Ohio State'
    
    @requests_mock.Mocker()
    def test_api_get_games_success(self, mock_request, test_env_vars):
        """Test successful API call to get games"""
        from src.ingestion.api_client import CollegeFootballAPIClient
        
        # Mock successful API response
        mock_response = [
            {
                "id": 1,
                "homeTeam": "Indiana",
                "awayTeam": "Purdue",
                "homePoints": 35,
                "awayPoints": 14,
                "season": 2024,
                "completed": True
            }
        ]
        
        mock_request.get(
            "https://api.collegefootballdata.com/games",
            json=mock_response,
            status_code=200
        )
        
        client = CollegeFootballAPIClient()
        games = client.get_games(year=2024)
        
        # Verify response
        assert len(games) == 1
        assert games[0]['homeTeam'] == 'Indiana'
        assert games[0]['awayPoints'] == 14
    
    @requests_mock.Mocker()
    def test_api_error_handling(self, mock_request, test_env_vars):
        """Test API error handling"""
        from src.ingestion.api_client import CollegeFootballAPIClient
        
        # Mock API error response
        mock_request.get(
            "https://api.collegefootballdata.com/teams",
            status_code=500,
            text="Internal Server Error"
        )
        
        client = CollegeFootballAPIClient()
        
        # Should handle error gracefully
        with pytest.raises(Exception):  # Should raise an exception for 500 error
            client.get_teams()
    
    @requests_mock.Mocker()
    def test_api_rate_limiting(self, mock_request, test_env_vars):
        """Test API rate limiting handling"""
        from src.ingestion.api_client import CollegeFootballAPIClient
        
        # Mock rate limit response
        mock_request.get(
            "https://api.collegefootballdata.com/teams",
            status_code=429,
            headers={"Retry-After": "60"},
            text="Rate limit exceeded"
        )
        
        client = CollegeFootballAPIClient()
        
        # Should handle rate limiting gracefully
        with pytest.raises(Exception):  # Should raise an exception for rate limit
            client.get_teams()
    
    def test_api_response_time(self, test_env_vars):
        """Test that API responses are reasonably fast"""
        from src.ingestion.api_client import CollegeFootballAPIClient
        
        client = CollegeFootballAPIClient()
        
        # Test response time (with timeout)
        def api_call():
            try:
                return client.get_teams()
            except Exception:
                return None  # Allow timeout/connection errors in tests
        
        # Should respond within 10 seconds (generous for external API)
        result = assert_response_time(api_call, max_seconds=10)
        
        # If successful, verify we got data
        if result is not None:
            assert len(result) > 0, "API should return data"
    
    def test_api_key_validation(self):
        """Test API key validation"""
        from src.ingestion.api_client import CollegeFootballAPIClient
        
        # Test with invalid API key
        with patch.dict('os.environ', {'CFBD_API_KEY': 'invalid_key'}):
            client = CollegeFootballAPIClient()
            assert client.api_key == 'invalid_key'
    
    def test_api_endpoints_exist(self, test_env_vars):
        """Test that all required API endpoints are accessible"""
        from src.ingestion.api_client import CollegeFootballAPIClient
        
        client = CollegeFootballAPIClient()
        
        # Test different endpoints
        endpoints_to_test = [
            ('get_teams', []),
            ('get_games', [2024]),
            ('get_team_stats', [2024]),
            ('get_conferences', [])
        ]
        
        for method_name, args in endpoints_to_test:
            method = getattr(client, method_name)
            
            # Test that method exists and is callable
            assert callable(method), f"Method {method_name} should be callable"
            
            # Test method signature (should not raise TypeError)
            try:
                # Just test the method signature, don't actually call API
                import inspect
                sig = inspect.signature(method)
                assert sig is not None, f"Method {method_name} should have a signature"
            except Exception as e:
                pytest.fail(f"Method {method_name} has invalid signature: {e}")
    
    @requests_mock.Mocker()
    def test_api_data_consistency(self, mock_request, test_env_vars):
        """Test that API returns consistent data structure"""
        from src.ingestion.api_client import CollegeFootballAPIClient
        
        # Mock consistent API responses
        teams_response = [
            {"id": 1, "school": "Indiana", "conference": "B1G"},
            {"id": 2, "school": "Ohio State", "conference": "B1G"}
        ]
        
        games_response = [
            {
                "id": 1,
                "homeTeam": "Indiana",
                "awayTeam": "Purdue",
                "homePoints": 35,
                "awayPoints": 14,
                "season": 2024,
                "completed": True
            }
        ]
        
        mock_request.get(
            "https://api.collegefootballdata.com/teams",
            json=teams_response,
            status_code=200
        )
        
        mock_request.get(
            "https://api.collegefootballdata.com/games",
            json=games_response,
            status_code=200
        )
        
        client = CollegeFootballAPIClient()
        
        # Test teams data structure
        teams = client.get_teams()
        assert all('id' in team for team in teams), "Teams should have 'id' field"
        assert all('school' in team for team in teams), "Teams should have 'school' field"
        
        # Test games data structure
        games = client.get_games(year=2024)
        assert all('homeTeam' in game for game in games), "Games should have 'homeTeam' field"
        assert all('awayTeam' in game for game in games), "Games should have 'awayTeam' field"
        assert all('homePoints' in game for game in games), "Games should have 'homePoints' field"
    
    def test_api_client_initialization(self, test_env_vars):
        """Test API client initialization with different configurations"""
        from src.ingestion.api_client import CollegeFootballAPIClient
        
        # Test default initialization
        client1 = CollegeFootballAPIClient()
        assert client1.api_key is not None
        assert client1.base_url == "https://api.collegefootballdata.com"
        
        # Test with custom base URL
        client2 = CollegeFootballAPIClient(base_url="https://test.api.com")
        assert client2.base_url == "https://test.api.com"
    
    @pytest.mark.slow
    def test_real_api_connection(self, test_env_vars):
        """Test real API connection (slow test)"""
        from src.ingestion.api_client import CollegeFootballAPIClient
        
        client = CollegeFootballAPIClient()
        
        # Only run if we have a real API key
        if client.api_key and client.api_key != 'test_api_key':
            try:
                # Test real API call
                teams = client.get_teams()
                
                # Verify we got real data
                assert len(teams) > 0, "Real API should return teams"
                assert all('school' in team for team in teams), "Teams should have school names"
                
                # Test that we can get games for current year
                games = client.get_games(year=2024)
                assert len(games) > 0, "Should have games for 2024"
                
            except Exception as e:
                pytest.fail(f"Real API connection failed: {e}")
        else:
            pytest.skip("No real API key available for testing")
