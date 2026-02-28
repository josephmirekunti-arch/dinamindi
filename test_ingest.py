import sys
from engine.store.db import DbStore
from engine.ingest.understat import UnderstatIngestor
from engine.validate.schema import MatchValidator

def main():
    print("Setting up database...")
    store = DbStore()
    store.create_all()
    session = store.get_session()
    
    print("Fetching matches for EPL 2023...")
    ingestor = UnderstatIngestor()
    matches = ingestor.fetch_season_matches('EPL', '2023')
    
    print(f"Fetched {len(matches)} matches from Understat API.")
    
    validator = MatchValidator()
    valid_matches = validator.validate_batch(matches)
    
    print(f"Validated {len(valid_matches)} matches.")
    
    # Save a few to DB to test
    for match in valid_matches[:5]:
        session.add(match)
        print(f"Added {match.home_team_name} vs {match.away_team_name} ({match.home_goals_ft}-{match.away_goals_ft})")
        
    session.commit()
    print("Committed to DB.")

if __name__ == "__main__":
    main()
