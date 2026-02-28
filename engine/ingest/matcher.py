import difflib

class TeamMatcher:
    """
    Utility to map API-Football team names to Understat team names using fuzzy matching.
    """
    
    # Pre-computed hard overrides for tricky names
    MANUAL_OVERRIDES = {
        # API-Football Name -> Understat Name
        "Manchester United": "Manchester United",
        "Newcastle": "Newcastle United",
        "Nottingham Forest": "Nottingham Forest",
        "Wolverhampton Wanderers": "Wolverhampton Wanderers",
        "Sheffield Utd": "Sheffield United",
        "Luton": "Luton",
        "Paris SG": "Paris Saint Germain",
        "Atletico Madrid": "Atletico Madrid",
        "Real Betis": "Real Betis",
        "Real Sociedad": "Real Sociedad",
        "Bayer Leverkusen": "Bayer Leverkusen",
        "Bayern Munich": "Bayern Munich",
        "Borussia Monchengladbach": "Borussia M.Gladbach",
        "Eintracht Frankfurt": "Eintracht Frankfurt"
    }

    @staticmethod
    def get_understat_name(api_name: str, known_understat_names: list[str]) -> str:
        """
        Takes an API-Football team name and attempts to find the closest match
        in the provided list of known Understat team names.
        """
        # 1. Check manual overrides first
        if api_name in TeamMatcher.MANUAL_OVERRIDES:
            mapped = TeamMatcher.MANUAL_OVERRIDES[api_name]
            if mapped in known_understat_names:
                return mapped
                
        # 2. Exact match check
        if api_name in known_understat_names:
            return api_name
            
        # 3. Fuzzy matching
        matches = difflib.get_close_matches(api_name, known_understat_names, n=1, cutoff=0.6)
        if matches:
            return matches[0]
            
        # 4. Fallback (return original and hope it works, or None if you prefer strict mapping)
        return api_name
