from engine.store.db import Match

class MatchValidator:
    def __init__(self):
        pass

    def validate_match(self, match: Match) -> bool:
        """
        Validates match integrity rules.
        """
        if not match.date_utc:
            return False
            
        if not match.home_team_id or not match.away_team_id:
            return False
            
        if match.home_team_id == match.away_team_id:
            return False
            
        if match.is_played:
            if match.home_goals_ft is None or match.away_goals_ft is None:
                return False
            if match.home_goals_ft < 0 or match.away_goals_ft < 0:
                return False
                
        return True

    def validate_batch(self, matches: list[Match]) -> list[Match]:
        """
        Returns only valid matches.
        """
        return [m for m in matches if self.validate_match(m)]
