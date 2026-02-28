from engine.ingest.understat import UnderstatIngestor
ingestor = UnderstatIngestor()
try:
    matches = ingestor.fetch_season_matches('La_liga', '2025')
    print(f"La Liga 2025 matches found: {len(matches)}")
except Exception as e:
    print(f"Error fetching La Liga: {e}")
