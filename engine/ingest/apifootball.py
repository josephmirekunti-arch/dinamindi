import os
import certifi
import aiohttp
import asyncio
from typing import List, Dict, Optional
import ssl

class APIFootballIngestor:
    """
    Client for interacting with the API-Football V3 REST API.
    Handles fetching live matches, historical statistics, and lineups.
    """
    
    BASE_URL = "https://v3.football.api-sports.io"
    
    # Mapping of our internal competition codes to API-Football league IDs
    LEAGUE_IDS = {
        'EPL': 39,
        'LALIGA': 140,
        'BUNDESLIGA': 78,
        'UCL': 2
    }
    
    def __init__(self, api_key: str = None):
        """
        Initialize the client. 
        Will look for 'API_FOOTBALL_KEY' in environment if not provided.
        """
        self.api_key = api_key or os.environ.get("API_FOOTBALL_KEY")
        if not self.api_key:
            raise ValueError("API-Football Key is missing! Set API_FOOTBALL_KEY environment variable.")
            
        self.headers = {
            "x-apisports-key": self.api_key,
            "x-apisports-host": "v3.football.api-sports.io"
        }
        
        # Use certifi for secure SSL verification
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    async def _fetch(self, session: aiohttp.ClientSession, endpoint: str, params: dict = None) -> dict:
        url = f"{self.BASE_URL}/{endpoint}"
        async with session.get(url, headers=self.headers, params=params, ssl=self.ssl_context) as response:
            if response.status != 200:
                print(f"API Error {response.status}: {await response.text()}")
                return {}
            data = await response.json()
            if data.get('errors'):
                print(f"API Football Error response: {data['errors']}")
            return data
            
    async def get_fixtures(self, league_code: str, season: str) -> List[dict]:
        """
        Fetch all fixtures for a specific league and season.
        Returns the raw fixture list JSON payload.
        """
        league_id = self.LEAGUE_IDS.get(league_code.upper())
        if not league_id:
            raise ValueError(f"Unknown league code: {league_code}")
            
        params = {
            "league": league_id,
            "season": season
        }
        
        all_fixtures = []
        async with aiohttp.ClientSession() as session:
            page = 1
            while True:
                params["page"] = page
                data = await self._fetch(session, "fixtures", params)
                responses = data.get('response', [])
                all_fixtures.extend(responses)
                
                paging = data.get('paging')
                if not paging or paging.get('current', 1) >= paging.get('total', 1):
                    break
                page += 1
                
            return all_fixtures
            
    async def get_fixture_statistics(self, fixture_id: int) -> List[dict]:
        """
        Fetch in-game statistics (possession, shots) for a specific match.
        """
        params = {"fixture": fixture_id}
        async with aiohttp.ClientSession() as session:
            data = await self._fetch(session, "fixtures/statistics", params)
            return data.get('response', [])
            
    async def get_fixture_lineups(self, fixture_id: int) -> List[dict]:
        """
        Fetch starting XIs and formations.
        """
        params = {"fixture": fixture_id}
        async with aiohttp.ClientSession() as session:
            data = await self._fetch(session, "fixtures/lineups", params)
            return data.get('response', [])
            
    async def get_fixture_odds(self, fixture_id: int, bookmaker_id: int = 8) -> List[dict]:
        """
        Fetch pre-match odds for a specific fixture. Defaults to Bet365 (8).
        """
        params = {
            "fixture": fixture_id,
            "bookmaker": bookmaker_id
        }
        async with aiohttp.ClientSession() as session:
            data = await self._fetch(session, "odds", params)
            return data.get('response', [])

    def fetch_season_sync(self, league_code: str, season: str) -> List[dict]:
        """
        Synchronous wrapper for downloading a full season of base fixture data.
        """
        return asyncio.run(self.get_fixtures(league_code, season))
