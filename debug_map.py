import asyncio
import os
import pandas as pd
from dotenv import load_dotenv
from engine.ingest.apifootball import APIFootballIngestor
from engine.ingest.matcher import TeamMatcher
from understatapi import UnderstatClient

load_dotenv()

async def debug_mapping():
    api_key = os.getenv('API_FOOTBALL_KEY')
    client = APIFootballIngestor(api_key)
    
    print("Fetching API-Football fixtures (EPL 2025)...")
    apif_runs = await client.get_fixtures('EPL', '2025')
    print(f"  - Total APIF matches: {len(apif_runs)}")
    
    print("Fetching Understat fixtures (EPL 2025)...")
    with UnderstatClient() as u_client:
        u_runs = u_client.league(league='EPL').get_match_data(season='2025')
        
    print(f"  - Total Understat matches: {len(u_runs)}")
    
    u_names = set([m['h']['title'] for m in u_runs])
    
    # Filter for unplayed matches (upcoming)
    upcoming_u = [m for m in u_runs if not m['isResult']]
    played_u = [m for m in u_runs if m['isResult']]
    
    print(f"  - Played: {len(played_u)}")
    print(f"  - Upcoming: {len(upcoming_u)}")
    
    print(f"\nComparing first 10 upcoming matches from Understat:")
    for m in upcoming_u[:10]:
        u_home = m['h']['title']
        u_date = m['datetime'][:10]
        u_key = (u_home, u_date)
        
        found = False
        potential_matches = []
        for fix in apif_runs:
            a_home_raw = fix['teams']['home']['name']
            a_home_matched = TeamMatcher.get_understat_name(a_home_raw, list(u_names))
            a_date = fix['fixture']['date'][:10]
            
            if a_home_matched == u_home:
                potential_matches.append(a_date)
                if a_date == u_date:
                    print(f"MATCH! {u_key} found in APIF (ID: {fix['fixture']['id']})")
                    found = True
                    break
        
        if not found:
            print(f"FAILED {u_key}")
            if potential_matches:
                print(f"  - Home team matched, but APIF dates are: {potential_matches}")
            else:
                print(f"  - No APIF fixture found for Home Team: {u_home}")

if __name__ == '__main__':
    asyncio.run(debug_mapping())
