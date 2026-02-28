import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import yaml
import numpy as np
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from engine.store.db import DbStore, Match
from engine.ingest.understat import UnderstatIngestor
from engine.validate.schema import MatchValidator
from engine.features.rolling import FeatureEngineer
from engine.model.poisson import PoissonGoalsModel
from engine.predict.probs import ProbabilityDeriver

app = FastAPI(title="Dinamindi - Football Probability Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production config: Allows Vercel frontends to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
if os.path.exists("static/assets"):
    app.mount("/assets", StaticFiles(directory="static/assets"), name="assets")

app.state.predict_data = []
app.state.performance_data = []
app.state.performance_summary = []
app.state.is_loaded = False
import threading
pipeline_lock = threading.Lock()

def load_pipeline():
    with pipeline_lock:
        if app.state.is_loaded:
            return
        print("Loading pipeline...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # Current Season with Pro API Plan
    target_season = '2025'
    competitions = config.get('competitions_in_scope', ['EPL', 'La_liga', 'Bundesliga'])
    
    # --- 1. Fetch Understat Base Data ---
    ingestor = UnderstatIngestor()
    matches = []
    for comp in competitions:
        print(f"Fetching Understat data for {comp}...")
        try:
            comp_matches = ingestor.fetch_season_matches(comp, target_season)
            matches.extend(comp_matches)
        except Exception as e:
            print(f"Failed to fetch Understat data for {comp}: {e}")
    
    validator = MatchValidator()
    valid_matches = validator.validate_batch(matches)
    
    # Extract Understat names
    understat_names = list(set([m.home_team_name for m in valid_matches] + [m.away_team_name for m in valid_matches]))
    
    # --- 2. Attempt to enrich with API-Football ---
    api_football_key = os.environ.get("API_FOOTBALL_KEY")
    api_football_raw = []
    
    if api_football_key:
        print(f"API_FOOTBALL_KEY detected. Enriching {target_season} data...")
        from engine.ingest.apifootball import APIFootballIngestor
        from engine.ingest.matcher import TeamMatcher
        
        api_client = APIFootballIngestor(api_key=api_football_key)
        try:
            import asyncio
            for comp in competitions:
                comp_key = comp.upper()
                if comp_key == 'LA_LIGA': comp_key = 'LALIGA'
                print(f"Fetching API-Football data for {comp_key}...")
                comp_raw = asyncio.run(api_client.get_fixtures(comp_key, target_season))
                api_football_raw.extend(comp_raw)
        except Exception as e:
            print(f"Failed to fetch API-Football data: {e}")
            
    # --- 3. Construct Unified Dataset ---
    import urllib.request
    import json
    
    # Sort matches chronologically
    sorted_matches = sorted(valid_matches, key=lambda x: x.date_utc)
    
    # Identify recently played matches for deep statistics enrichment
    # With 7,500 requests/day, we can now afford to fetch stats for a larger history (e.g., last 100 matches)
    recent_played_ids = [m.match_id for m in sorted_matches if m.is_played][-100:]
    
    df_data = []
    for m in sorted_matches:
        base_record = {
            'id': m.match_id, 'date_utc': m.date_utc, 'competition': m.competition, 'season': m.season,
            'home_team_id': m.home_team_id, 'away_team_id': m.away_team_id, 'home_team_name': m.home_team_name, 'away_team_name': m.away_team_name,
            'home_goals_ft': m.home_goals_ft, 'away_goals_ft': m.away_goals_ft, 'home_xg': m.home_xg, 'away_xg': m.away_xg,
            'is_played': m.is_played,
            'apifootball_id': None, 'home_possession': None, 'away_possession': None, 'home_sot': None, 'away_sot': None,
            'home_logo': None, 'away_logo': None
        }
        
        # Robust Mapping Strategy
        m_date = m.date_utc.date()
        best_fix = None
        for fix in api_football_raw:
            a_home_raw = fix['teams']['home']['name']
            a_date = datetime.strptime(fix['fixture']['date'][:10], "%Y-%m-%d").date()
            if abs((a_date - m_date).days) <= 2:
                a_home_matched = TeamMatcher.get_understat_name(a_home_raw, understat_names)
                if a_home_matched == m.home_team_name:
                    best_fix = fix
                    break
        
        if best_fix:
            base_record['apifootball_id'] = best_fix['fixture']['id']
            base_record['home_logo'] = best_fix.get('teams', {}).get('home', {}).get('logo')
            base_record['away_logo'] = best_fix.get('teams', {}).get('away', {}).get('logo')
            
            # --- REAL-TIME MATCH COMPLETION OVERRIDE ---
            # Understat is slow to update. If API-Football says it's done, trust it.
            status_short = best_fix['fixture']['status']['short']
            if status_short in ['FT', 'AET', 'PEN']:
                base_record['is_played'] = True
                if best_fix['goals']['home'] is not None:
                    base_record['home_goals_ft'] = int(best_fix['goals']['home'])
                if best_fix['goals']['away'] is not None:
                    base_record['away_goals_ft'] = int(best_fix['goals']['away'])
                    
            # Enrichment: Fetch Possession, SOT, and Goal Events (Pro API capacity)
            # Fetch if played AND (was in Understat recent list OR played in the last 7 days)
            is_recent = m.match_id in recent_played_ids or abs((datetime.utcnow().date() - m_date).days) <= 7
            
            if base_record['is_played'] and is_recent and api_football_key:
                try:
                    # 1. Fetch Basic Statistics (Possession/SOT)
                    req = urllib.request.Request(
                        f"https://v3.football.api-sports.io/fixtures/statistics?fixture={base_record['apifootball_id']}", 
                        headers={'x-apisports-key': api_football_key}
                    )
                    with urllib.request.urlopen(req) as resp:
                        stats_data = json.loads(resp.read().decode()).get('response', [])
                    if stats_data and len(stats_data) >= 2:
                        h_stats = stats_data[0]['statistics']
                        a_stats = stats_data[1]['statistics']
                        def _get(a, k):
                            for s in a:
                                if s['type'] == k and s['value'] is not None:
                                    val = str(s['value']).replace('%', '') if s['value'] is not None else "0"
                                    return int(val)
                            return 0
                        base_record['home_possession'] = _get(h_stats, 'Ball Possession')
                        base_record['away_possession'] = _get(a_stats, 'Ball Possession')
                        base_record['home_sot'] = _get(h_stats, 'Shots on Goal')
                        base_record['away_sot'] = _get(a_stats, 'Shots on Goal')
                        base_record['home_corners'] = _get(h_stats, 'Corner Kicks')
                        base_record['away_corners'] = _get(a_stats, 'Corner Kicks')
                        base_record['home_yellows'] = _get(h_stats, 'Yellow Cards')
                        base_record['away_yellows'] = _get(a_stats, 'Yellow Cards')
                        base_record['home_reds'] = _get(h_stats, 'Red Cards')
                        base_record['away_reds'] = _get(a_stats, 'Red Cards')
                    
                    # 2. Fetch Goal Events for Interval Analysis
                    req_ev = urllib.request.Request(
                        f"https://v3.football.api-sports.io/fixtures/events?fixture={base_record['apifootball_id']}&type=goal", 
                        headers={'x-apisports-key': api_football_key}
                    )
                    with urllib.request.urlopen(req_ev) as resp:
                        ev_data = json.loads(resp.read().decode()).get('response', [])
                    
                    goals = []
                    for ev in ev_data:
                        if ev.get('type') == 'Goal':
                            goals.append({
                                'team': 'home' if ev['team']['id'] == best_fix['teams']['home']['id'] else 'away',
                                'minute': ev['time']['elapsed'],
                                'extra': ev['time']['extra']
                            })
                    base_record['goal_events'] = json.dumps(goals)
                except Exception as e:
                    print(f"Stats/Events enrichment failed for {m.home_team_name}: {e}")
                
        df_data.append(base_record)
    
    matches_df = pd.DataFrame(df_data)
    print(f"Successfully mapped {matches_df['apifootball_id'].notnull().sum()} matches with Pro enrichment.")
    
    fe = FeatureEngineer(
        windows=config.get('rolling_windows', [5, 10, 20]),
        max_rest_days=14
    )
    features_df = fe.compute_features(matches_df)
    feature_cols = [c for c in features_df.columns if c.startswith('delta_')]
    features_df = features_df.dropna(subset=feature_cols)
    
    train_df = features_df[features_df['is_played'] == True].copy()
    predict_df = features_df[features_df['is_played'] == False].copy()
    
    if len(train_df) < 20:
        return
        
    model = PoissonGoalsModel(alpha=1.0)
    weights = np.ones(len(train_df))
    model.fit(train_df, team_cols=[], covariate_cols=feature_cols, weights=weights)
    
    deriver = ProbabilityDeriver(max_goals=config.get('max_goals', 8))
    thresholds = config.get('selection_thresholds', {})
    
    # Process Predictions (Upcoming)
    predict_data = []
    if len(predict_df) > 0:
        preds = model.predict_lambdas(predict_df, feature_cols)
        predict_df['lambda_h'] = preds['lambda_h'].values
        predict_df['lambda_a'] = preds['lambda_a'].values
        predict_df = predict_df.sort_values(by='date_utc')
        
        for idx, row in predict_df.head(15).iterrows():
            mkt = deriver.derive_markets(row['lambda_h'], row['lambda_a'])
            
            # Fetch Bookmaker Odds for EV comparison
            fixture_odds = {}
            if api_football_key and pd.notnull(row.get('apifootball_id')):
                try:
                    # Try Bet365 first (Bookmaker 8)
                    odds_url = f"https://v3.football.api-sports.io/odds?fixture={int(row['apifootball_id'])}&bookmaker=8"
                    req_odds = urllib.request.Request(odds_url, headers={'x-apisports-key': api_football_key})
                    with urllib.request.urlopen(req_odds) as resp:
                        odds_res = json.loads(resp.read().decode()).get('response', [])
                    
                    # Fallback to any bookmaker if Bet365 is missing
                    if not odds_res:
                        odds_url = f"https://v3.football.api-sports.io/odds?fixture={int(row['apifootball_id'])}"
                        req_odds = urllib.request.Request(odds_url, headers={'x-apisports-key': api_football_key})
                        with urllib.request.urlopen(req_odds) as resp:
                            odds_res = json.loads(resp.read().decode()).get('response', [])
                            
                    if odds_res:
                        bookies = odds_res[0].get('bookmakers', [])
                        if bookies:
                            # Take first available bookmaker
                            bets = bookies[0].get('bets', [])
                            for bet in bets:
                                if bet['name'] in ['Match Winner', 'Full Time Result']:
                                    for v in bet['values']:
                                        if v['value'] == 'Home': fixture_odds['1X2 HOME WINS'] = float(v['odd'])
                                        elif v['value'] == 'Away': fixture_odds['1X2 AWAY WINS'] = float(v['odd'])
                                        elif v['value'] == 'Draw': fixture_odds['1X2 DRAW'] = float(v['odd'])
                                elif bet['name'] == 'Goals Over/Under':
                                    for v in bet['values']:
                                        if v['value'] == 'Over 2.5': fixture_odds['OVER 2.5'] = float(v['odd'])
                                        elif v['value'] == 'Under 2.5': fixture_odds['UNDER 2.5'] = float(v['odd'])
                                elif bet['name'] == 'Both Teams Score':
                                    for v in bet['values']:
                                        if v['value'] == 'Yes': fixture_odds['BTTS YES'] = float(v['odd'])
                                        elif v['value'] == 'No': fixture_odds['BTTS NO'] = float(v['odd'])
                except Exception as e:
                    print(f"Odds fetch failed for {row['home_team_name']}: {e}")
                    
            # Secondary Markets
            # Expected Match Totals = Home Rolling For + Away Rolling For
            h_corn = row.get('home_roll_corners_for_5', 4.5)
            a_corn = row.get('away_roll_corners_for_5', 4.5)
            h_card = row.get('home_roll_cards_for_5', 2.0)
            a_card = row.get('away_roll_cards_for_5', 2.0)
            sec_recs = deriver.derive_secondary_markets(h_corn + a_corn, h_card + a_card)
            
            recs = deriver.get_structured_recommendations(mkt, thresholds, row['lambda_h'], row['lambda_a'], fixture_odds)
            recs.extend(sec_recs)
            
            # Elo Insights
            h_elo = row.get('home_elo_pre', 1500)
            a_elo = row.get('away_elo_pre', 1500)
            elo_insights = deriver.derive_elo_insights(h_elo, a_elo, row['lambda_h'], row['lambda_a'])
            
            # History for interval stats (Pro Plan data available in features_df)
            h_hist = train_df[train_df['home_team_name'] == row['home_team_name']]
            a_hist = train_df[train_df['away_team_name'] == row['away_team_name']]
            intervals = deriver.get_interval_stats(h_hist, a_hist)
            
            h_logo = row.get('home_logo')
            a_logo = row.get('away_logo')
            
            predict_data.append({
                "date": row['date_utc'].isoformat(),
                "competition": row['competition'],
                "home_team": row['home_team_name'],
                "away_team": row['away_team_name'],
                "home_logo": str(h_logo) if pd.notnull(h_logo) else None,
                "away_logo": str(a_logo) if pd.notnull(a_logo) else None,
                "lambda_h": round(row['lambda_h'], 2),
                "lambda_a": round(row['lambda_a'], 2),
                "home_elo": round(h_elo),
                "away_elo": round(a_elo),
                "elo_insights": elo_insights,
                "markets": recs,
                "goal_intervals": intervals
            })
            
    # Process Performance (Past)
    # We will score the last 20 played matches to see how the model performed
    performance_data = []
    recent_train = train_df.sort_values(by='date_utc', ascending=False).head(20)
    
    if len(recent_train) > 0:
        perf_preds = model.predict_lambdas(recent_train, feature_cols)
        recent_train['lambda_h'] = perf_preds['lambda_h'].values
        recent_train['lambda_a'] = perf_preds['lambda_a'].values
        
        for idx, row in recent_train.iterrows():
            mkt = deriver.derive_markets(row['lambda_h'], row['lambda_a'])
            
            # Determine actual results
            hg = row['home_goals_ft']
            ag = row['away_goals_ft']
            total_goals = hg + ag
            
            # Note: For historical games, API-Football data (corners/yellows) might not be in the row if the match was scraped before integration.
            # We use .get with a default of None to avoid KeyErrors.
            hc = row.get('home_corners', None)
            ac = row.get('away_corners', None)
            total_corners = hc + ac if hc is not None and ac is not None and not np.isnan(hc) and not np.isnan(ac) else None
            
            hy = row.get('home_yellows', None)
            ay = row.get('away_yellows', None)
            hr = row.get('home_reds', None)
            ar = row.get('away_reds', None)
            
            total_cards = None
            if all(v is not None and not np.isnan(v) for v in [hy, ay, hr, ar]):
                total_cards = hy + ay + hr + ar
            
            actual_1x2 = 'HOME' if hg > ag else ('AWAY' if ag > hg else 'DRAW')
            actual_o25 = total_goals > 2.5
            actual_btts = hg > 0 and ag > 0
            
            # See what the model recommended
            recs = deriver.get_structured_recommendations(mkt, thresholds, row['lambda_h'], row['lambda_a'])
            
            # Add secondary market recommendations for the performance check
            # We use the same baseline logic as the predictions endpoint
            h_corn = row.get('home_roll_corners_for_5', 4.5)
            a_corn = row.get('away_roll_corners_for_5', 4.5)
            h_card = row.get('home_roll_cards_for_5', 2.0)
            a_card = row.get('away_roll_cards_for_5', 2.0)
            sec_recs = deriver.derive_secondary_markets(h_corn + a_corn, h_card + a_card)
            recs.extend(sec_recs)
            
            recommended_markets = [r['market'] for r in recs if r['risk'] == 'recommended']
            
            # Check hits
            hits = []
            metrics = []
            
            for rec_mkt in recommended_markets:
                hit = False
                unverifiable = False
                
                # Primary Markets
                if '1X2 HOME' in rec_mkt and actual_1x2 == 'HOME': hit = True
                if '1X2 AWAY' in rec_mkt and actual_1x2 == 'AWAY': hit = True
                if 'OVER 2.5' in rec_mkt and actual_o25: hit = True
                if 'UNDER 2.5' in rec_mkt and not actual_o25: hit = True
                if 'BTTS YES' in rec_mkt and actual_btts: hit = True
                if 'BTTS NO' in rec_mkt and not actual_btts: hit = True
                if 'OVER 1.5' in rec_mkt and total_goals > 1.5: hit = True
                if 'UNDER 1.5' in rec_mkt and total_goals < 1.5: hit = True
                if 'OVER 0.5' in rec_mkt and total_goals > 0.5: hit = True
                if 'UNDER 0.5' in rec_mkt and total_goals < 0.5: hit = True
                
                # Secondary Markets (Only evaluate if we actually have the data for this historic match)
                if 'CORNERS' in rec_mkt:
                    if total_corners is not None:
                        if 'OVER 8.5' in rec_mkt and total_corners > 8.5: hit = True
                        if 'UNDER 8.5' in rec_mkt and total_corners < 8.5: hit = True
                        if 'OVER 9.5' in rec_mkt and total_corners > 9.5: hit = True
                        if 'UNDER 9.5' in rec_mkt and total_corners < 9.5: hit = True
                    else:
                        unverifiable = True
                        
                if 'CARDS' in rec_mkt:
                    if total_cards is not None:
                        if 'OVER 3.5' in rec_mkt and total_cards > 3.5: hit = True
                        if 'UNDER 3.5' in rec_mkt and total_cards < 3.5: hit = True
                        if 'OVER 4.5' in rec_mkt and total_cards > 4.5: hit = True
                        if 'UNDER 4.5' in rec_mkt and total_cards < 4.5: hit = True
                    else:
                        unverifiable = True

                # Assign status
                if unverifiable:
                    status = "PENDING"  # Too old to verify
                else:
                    status = "WON" if hit else "LOST"
                    
                metrics.append({"market": rec_mkt, "status": status})
                
            h_logo_perf = row.get('home_logo')
            a_logo_perf = row.get('away_logo')
                
            performance_data.append({
                "date": row['date_utc'].isoformat(),
                "competition": row['competition'],
                "home_team": row['home_team_name'],
                "away_team": row['away_team_name'],
                "home_logo": str(h_logo_perf) if pd.notnull(h_logo_perf) else None,
                "away_logo": str(a_logo_perf) if pd.notnull(a_logo_perf) else None,
                "score": f"{int(hg)} - {int(ag)}",
                "recommended": metrics
            })
            
    summary_counts = {}
    for match in performance_data[:10]:
        for rec in match['recommended']:
            mkt = rec['market']
            is_won = 1 if rec['status'] == 'WON' else 0
            if mkt not in summary_counts:
                summary_counts[mkt] = {"won": 0, "total": 0}
            summary_counts[mkt]["won"] += is_won
            summary_counts[mkt]["total"] += 1
            
    summary_list = []
    for mkt, counts in summary_counts.items():
        if counts["total"] > 0:
            acc = (counts["won"] / counts["total"]) * 100.0
            summary_list.append({
                "market": mkt,
                "accuracy": round(acc, 1),
                "won": counts["won"],
                "total": counts["total"]
            })
            
    summary_list.sort(key=lambda x: (-x["accuracy"], -x["total"]))
            
    app.state.predict_data = predict_data
    app.state.performance_data = performance_data
    app.state.performance_summary = summary_list
    app.state.is_loaded = True
    print("Pipeline loaded.")

import asyncio
import urllib.request
import json
from datetime import datetime, timedelta

async def poll_live_matches():
    while True:
        try:
            api_football_key = os.environ.get("API_FOOTBALL_KEY")
            if app.state.is_loaded and app.state.predict_data and api_football_key:
                now = datetime.utcnow()
                live_matches = []
                # Find matches that started at least 90 mins ago, up to 4 hours ago 
                # Or just any match that is past its kick-off time and not in performance_data
                for m in app.state.predict_data:
                    m_date = datetime.fromisoformat(m['date'])
                    if m_date <= now:
                        live_matches.append(m)
                
                for match in live_matches:
                    fix_id = match.get('apifootball_id')
                    if not fix_id:
                        continue
                        
                    # Ping API-Football
                    req = urllib.request.Request(
                        f"https://v3.football.api-sports.io/fixtures?id={fix_id}", 
                        headers={'x-apisports-key': api_football_key}
                    )
                    with urllib.request.urlopen(req) as resp:
                        fix_data = json.loads(resp.read().decode()).get('response', [])
                        
                    if fix_data:
                        status_short = fix_data[0]['fixture']['status']['short']
                        if status_short in ['FT', 'AET', 'PEN']:
                            print(f"[LIVE POLLER] Match Completed: {match['home_team']} vs {match['away_team']}")
                            
                            hg = fix_data[0]['goals']['home']
                            ag = fix_data[0]['goals']['away']
                            
                            # Fetch Stats for Secondary Markets
                            req_stats = urllib.request.Request(
                                f"https://v3.football.api-sports.io/fixtures/statistics?fixture={fix_id}", 
                                headers={'x-apisports-key': api_football_key}
                            )
                            with urllib.request.urlopen(req_stats) as resp:
                                stats_data = json.loads(resp.read().decode()).get('response', [])
                            
                            hc, ac, hy, ay, hr, ar = None, None, None, None, None, None
                            if stats_data and len(stats_data) >= 2:
                                def _get(a, k):
                                    for s in a:
                                        if s['type'] == k and s['value'] is not None:
                                            return int(str(s['value']).replace('%', ''))
                                    return 0
                                hc = _get(stats_data[0]['statistics'], 'Corner Kicks')
                                ac = _get(stats_data[1]['statistics'], 'Corner Kicks')
                                hy = _get(stats_data[0]['statistics'], 'Yellow Cards')
                                ay = _get(stats_data[1]['statistics'], 'Yellow Cards')
                                hr = _get(stats_data[0]['statistics'], 'Red Cards')
                                ar = _get(stats_data[1]['statistics'], 'Red Cards')
                                
                            total_goals = hg + ag
                            total_corners = hc + ac if hc is not None and ac is not None else None
                            total_cards = hy + ay + hr + ar if hy is not None and ay is not None and hr is not None and ar is not None else None
                            
                            actual_1x2 = 'HOME' if hg > ag else ('AWAY' if ag > hg else 'DRAW')
                            actual_o25 = total_goals > 2.5
                            actual_btts = hg > 0 and ag > 0
                            
                            # Evaluate hit/miss
                            metrics = []
                            for rec in match.get('markets', []):
                                if rec.get('risk') == 'recommended':
                                    rec_mkt = rec['market']
                                    hit = False
                                    unverifiable = False
                                    if '1X2 HOME' in rec_mkt and actual_1x2 == 'HOME': hit = True
                                    if '1X2 AWAY' in rec_mkt and actual_1x2 == 'AWAY': hit = True
                                    if 'OVER 2.5' in rec_mkt and actual_o25: hit = True
                                    if 'UNDER 2.5' in rec_mkt and not actual_o25: hit = True
                                    if 'BTTS YES' in rec_mkt and actual_btts: hit = True
                                    if 'BTTS NO' in rec_mkt and not actual_btts: hit = True
                                    if 'OVER 1.5' in rec_mkt and total_goals > 1.5: hit = True
                                    if 'UNDER 1.5' in rec_mkt and total_goals < 1.5: hit = True
                                    if 'OVER 0.5' in rec_mkt and total_goals > 0.5: hit = True
                                    if 'UNDER 0.5' in rec_mkt and total_goals < 0.5: hit = True
                                    if 'CORNERS' in rec_mkt:
                                        if total_corners is not None:
                                            if 'OVER 8.5' in rec_mkt and total_corners > 8.5: hit = True
                                            if 'UNDER 8.5' in rec_mkt and total_corners < 8.5: hit = True
                                            if 'OVER 9.5' in rec_mkt and total_corners > 9.5: hit = True
                                            if 'UNDER 9.5' in rec_mkt and total_corners < 9.5: hit = True
                                        else: unverifiable = True
                                    if 'CARDS' in rec_mkt:
                                        if total_cards is not None:
                                            if 'OVER 3.5' in rec_mkt and total_cards > 3.5: hit = True
                                            if 'UNDER 3.5' in rec_mkt and total_cards < 3.5: hit = True
                                            if 'OVER 4.5' in rec_mkt and total_cards > 4.5: hit = True
                                            if 'UNDER 4.5' in rec_mkt and total_cards < 4.5: hit = True
                                        else: unverifiable = True
                                        
                                    status = "PENDING" if unverifiable else ("WON" if hit else "LOST")
                                    metrics.append({"market": rec_mkt, "status": status})
                                    
                            perf_entry = {
                                "date": match['date'],
                                "competition": match.get('competition'),
                                "home_team": match['home_team'],
                                "away_team": match['away_team'],
                                "home_logo": match.get('home_logo'),
                                "away_logo": match.get('away_logo'),
                                "score": f"{int(hg)} - {int(ag)}",
                                "recommended": metrics
                            }
                            
                            # Migrate safely with the lock
                            with pipeline_lock:
                                app.state.performance_data.insert(0, perf_entry)
                                app.state.predict_data = [x for x in app.state.predict_data if x.get('apifootball_id') != fix_id]
                                
                                # Recalculate summary
                                summary_counts = {}
                                for perf in app.state.performance_data[:10]:
                                    for rec_perf in perf['recommended']:
                                        mkt_perf = rec_perf['market']
                                        if mkt_perf not in summary_counts: summary_counts[mkt_perf] = {"won": 0, "total": 0}
                                        summary_counts[mkt_perf]["won"] += 1 if rec_perf['status'] == 'WON' else 0
                                        summary_counts[mkt_perf]["total"] += 1
                                        
                                summary_list = []
                                for m_name, counts in summary_counts.items():
                                    if counts["total"] > 0:
                                        acc = (counts["won"] / counts["total"]) * 100.0
                                        summary_list.append({"market": m_name, "accuracy": round(acc, 1), "won": counts["won"], "total": counts["total"]})
                                summary_list.sort(key=lambda x: (-x["accuracy"], -x["total"]))
                                app.state.performance_summary = summary_list
        except Exception as e:
            print(f"[LIVE POLLER ERROR] {e}")
            
        await asyncio.sleep(900)  # Check every 15 minutes

@app.on_event("startup")
async def startup_event():
    import threading
    # Run the heavy pipeline sync load in a background thread so as not to block app startup entirely
    # But wait, we want predictions available. We'll let load_pipeline run lazily on first request or pre-emptively here.
    asyncio.create_task(poll_live_matches())

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/api/predictions")
def get_predictions():
    if not app.state.is_loaded:
        load_pipeline()
    return {"data": app.state.predict_data}
    
@app.get("/api/performance")
def get_performance():
    if not app.state.is_loaded:
        load_pipeline()
    return {
        "data": app.state.performance_data,
        "summary": app.state.performance_summary
    }

if __name__ == "__main__":
    import threading
    threading.Thread(target=load_pipeline, daemon=True).start()
    
    # Production config: Cloud platforms (Render, Railway, Heroku) inject the $PORT env var
    port = int(os.environ.get("PORT", 8004))
    uvicorn.run(app, host="0.0.0.0", port=port)
