import asyncio
from datetime import datetime
from understatapi import UnderstatClient
from engine.store.db import Match

class UnderstatIngestor:
    COMPETITION_MAP = {
        'EPL': 'EPL',
        'La_Liga': 'La_Liga',
        'Bundesliga': 'Bundesliga'
    }

    def __init__(self):
        # We use understatapi to fetch the matches
        pass

    def fetch_season_matches(self, league: str, season: str):
        """
        Fetches match data for a specific league and season using understatapi.
        league must be one of: EPL, La_liga, Bundesliga
        season must be a string year, e.g., '2023' for the 23/24 season.
        """
        if league not in self.COMPETITION_MAP.values():
            raise ValueError(f"League {league} not supported. Must be one of {list(self.COMPETITION_MAP.values())}")
            
        with UnderstatClient() as client:
            league_match_data = client.league(league=league).get_match_data(season=season)
            
        return self._parse_matches(league_match_data, league, int(season))

    def _parse_matches(self, raw_data: list, competition: str, season: int) -> list[Match]:
        """
        Parses the raw JSON payload from Understat into SQLAlchemy Match objects.
        """
        matches = []
        for row in raw_data:
            # Understat match IDs
            match_id = str(row['id'])
            
            # Date
            date_utc = datetime.strptime(row['datetime'], "%Y-%m-%d %H:%M:%S")
            
            # Teams
            home_team_id = str(row['h']['id'])
            away_team_id = str(row['a']['id'])
            home_team_name = row['h']['title']
            away_team_name = row['a']['title']
            
            # Goals
            # For unplayed matches, these will be None
            try:
                home_goals = int(row['goals']['h']) if row['goals']['h'] is not None else None
                away_goals = int(row['goals']['a']) if row['goals']['a'] is not None else None
            except (ValueError, TypeError):
                home_goals = None
                away_goals = None
                
            # xG
            try:
                home_xg = float(row['xG']['h']) if row['xG']['h'] is not None else None
                away_xg = float(row['xG']['a']) if row['xG']['a'] is not None else None
            except (ValueError, TypeError):
                home_xg = None
                away_xg = None
            
            is_played = row['isResult']
            
            # If not played, we still might want the fixture to generate predictions later
            match = Match(
                match_id=match_id,
                date_utc=date_utc,
                competition=competition,
                season=season,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                home_team_name=home_team_name,
                away_team_name=away_team_name,
                home_goals_ft=home_goals,
                away_goals_ft=away_goals,
                home_xg=home_xg,
                away_xg=away_xg,
                is_played=is_played
            )
            matches.append(match)
            
        return matches
