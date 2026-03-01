import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, windows=[5, 10, 20], max_rest_days=14):
        self.windows = windows
        self.max_rest_days = max_rest_days

    def _compute_elo(self, matches_df: pd.DataFrame, squad_churn: dict = None) -> pd.DataFrame:
        df = matches_df.sort_values('date_utc').copy()
        
        elo_dict = {}
        team_last_season = {}
        
        home_elo_pre = []
        away_elo_pre = []
        
        for idx, row in df.iterrows():
            h_id = row['home_team_id']
            a_id = row['away_team_id']
            season = row['season']
            
            # Initialize or retrieve ratings
            r_h = elo_dict.get(h_id, 1500.0)
            r_a = elo_dict.get(a_id, 1500.0)
            
            # Season reversion with Roster Volatility Check
            # research recommendation: reset more aggressively if squad churn >30%
            if team_last_season.get(h_id, season) != season:
                churn = squad_churn.get(h_id, 0.2) if squad_churn else 0.2
                # If churn > 0.3, we reset by 40% instead of 20%
                reg_factor = 0.6 if churn > 0.3 else 0.8
                r_h = r_h * reg_factor + 1500.0 * (1.0 - reg_factor)
            if team_last_season.get(a_id, season) != season:
                churn = squad_churn.get(a_id, 0.2) if squad_churn else 0.2
                reg_factor = 0.6 if churn > 0.3 else 0.8
                r_a = r_a * reg_factor + 1500.0 * (1.0 - reg_factor)
                
            team_last_season[h_id] = season
            team_last_season[a_id] = season
            
            home_elo_pre.append(r_h)
            away_elo_pre.append(r_a)
            
            # If the match hasn't been played (no goals), record pre-match Elo and don't update
            if pd.isna(row['home_goals_ft']) or pd.isna(row['away_goals_ft']):
                continue
                
            gh = row['home_goals_ft']
            ga = row['away_goals_ft']
            
            # Expected score, Home Advantage = +80
            e_h = 1.0 / (1.0 + 10.0 ** ((r_a - (r_h + 80.0)) / 400.0))
            
            # Match outcome
            if gh > ga:
                s_h = 1.0
            elif gh < ga:
                s_h = 0.0
            else:
                s_h = 0.5
                
            # Goal Margin Multiplier (GMM)
            gd = abs(gh - ga)
            if gd <= 1:
                gmm = 1.0
            elif gd == 2:
                gmm = 1.5
            else:
                gmm = (11.0 + gd) / 8.0
                
            # Update (K=20)
            shift = 20.0 * gmm * (s_h - e_h)
            
            elo_dict[h_id] = r_h + shift
            elo_dict[a_id] = r_a - shift
            
        df['home_elo_pre'] = home_elo_pre
        df['away_elo_pre'] = away_elo_pre
        df['delta_elo'] = df['home_elo_pre'] - df['away_elo_pre']
        return df

    def compute_features(self, matches_df: pd.DataFrame, squad_churn: dict = None, top_players: dict = None, player_ratings: dict = None, manager_changes: dict = None) -> pd.DataFrame:
        """
        Takes a DataFrame of historical matches, chronologically sorted, 
        and computes rolling window features for home and away teams.
        Includes Model V2 features: Player-Level Ratings, Managerial Change, and Travel Distance.
        """
        import json
        
        # 1. Compute Elo with Roster Volatility
        matches_df = self._compute_elo(matches_df, squad_churn)
        
        # 2. Player Outlier Tracking & Starting XI Ratings
        def process_lineup(row, team_prefix='home'):
            team_id = row[f'{team_prefix}_team_id']
            lineup_str = row.get(f'{team_prefix}_lineup')
            outliers_count = 0
            xi_rating = 1500.0 # Baseline Elo-like rating for players
            
            if lineup_str:
                try:
                    lineup = json.loads(lineup_str)
                    
                    # Top Outliers Count
                    if top_players and str(team_id) in top_players:
                        stars = top_players[str(team_id)]
                        for player in lineup:
                            p_id = player.get('id') or player.get('name')
                            if p_id in stars:
                                outliers_count += 1
                                
                    # Aggregate Player-Level Ratings
                    if player_ratings:
                        ratings = []
                        for player in lineup:
                            p_id = player.get('id')
                            if p_id and str(p_id) in player_ratings:
                                ratings.append(player_ratings[str(p_id)])
                        if ratings:
                            xi_rating = np.mean(ratings)
                except:
                    pass
                    
            return pd.Series([outliers_count, xi_rating])

        matches_df[['home_outliers', 'home_xi_rating']] = matches_df.apply(lambda r: process_lineup(r, 'home'), axis=1)
        matches_df[['away_outliers', 'away_xi_rating']] = matches_df.apply(lambda r: process_lineup(r, 'away'), axis=1)
        matches_df['delta_outliers'] = matches_df['home_outliers'] - matches_df['away_outliers']
        matches_df['delta_xi_rating'] = matches_df['home_xi_rating'] - matches_df['away_xi_rating']
        
        # 3. Managerial Change "Shock Therapy" Flag (Last 30 days)
        def check_manager_bounce(row, team_prefix='home'):
            if not manager_changes: return 0
            team_id = row[f'{team_prefix}_team_id']
            match_date = pd.to_datetime(row['date_utc']).tz_localize(None)
            
            changes = manager_changes.get(str(team_id), [])
            for change_date_str in changes:
                change_date = pd.to_datetime(change_date_str).tz_localize(None)
                days_since = (match_date - change_date).days
                if 0 <= days_since <= 30: # 30-day shock therapy window
                    return 1
            return 0
            
        matches_df['home_manager_bounce'] = matches_df.apply(lambda r: check_manager_bounce(r, 'home'), axis=1)
        matches_df['away_manager_bounce'] = matches_df.apply(lambda r: check_manager_bounce(r, 'away'), axis=1)
        
        # 4. Environmental Variables (Placeholder calculated externally or defaulted)
        # Using a default distance 0. In a full pipeline, we'd use Haversine based on stadium coords.
        if 'travel_distance' not in matches_df.columns:
            matches_df['travel_distance'] = 0.0

        # Ensure we are sorted by date
        matches_df = matches_df.sort_values(by='date_utc').reset_index(drop=True)
        
        # Optional Advanced Stats
        if 'home_xt' not in matches_df.columns:
            # Proxy xT based on Shot volume + Possession until API integration
            matches_df['home_xt'] = (matches_df['home_xg'].fillna(1.0) * matches_df['home_possession'].fillna(50) / 50.0) * 0.8
            matches_df['away_xt'] = (matches_df['away_xg'].fillna(1.0) * matches_df['away_possession'].fillna(50) / 50.0) * 0.8

        # Melt the dataframe so each row is a team-match with context
        home_cols = ['id', 'date_utc', 'competition', 'season', 'home_team_id', 'home_goals_ft', 'away_goals_ft', 'home_xg', 'away_xg', 'home_xt', 'away_xt', 'home_possession', 'away_possession', 'home_sot', 'away_sot', 'home_corners', 'away_corners', 'home_outliers']
        away_cols = ['id', 'date_utc', 'competition', 'season', 'away_team_id', 'away_goals_ft', 'home_goals_ft', 'away_xg', 'home_xg', 'away_xt', 'home_xt', 'away_possession', 'home_possession', 'away_sot', 'home_sot', 'away_corners', 'home_corners', 'away_outliers']
        
        # Handle missing columns initially
        for col in home_cols + away_cols:
            if col not in matches_df.columns:
                matches_df[col] = np.nan
        
        home_df = matches_df[home_cols].copy()
        home_df['is_home'] = 1
        home_df.columns = ['match_row_id', 'date_utc', 'competition', 'season', 'team_id', 'gf', 'ga', 'xg_for', 'xg_against', 'xt_for', 'xt_against', 'poss_for', 'poss_against', 'sot_for', 'sot_against', 'corners_for', 'corners_against', 'outliers', 'is_home']
        
        away_df = matches_df[away_cols].copy()
        away_df['is_home'] = 0
        away_df.columns = ['match_row_id', 'date_utc', 'competition', 'season', 'team_id', 'gf', 'ga', 'xg_for', 'xg_against', 'xt_for', 'xt_against', 'poss_for', 'poss_against', 'sot_for', 'sot_against', 'corners_for', 'corners_against', 'outliers', 'is_home']
        
        team_matches = pd.concat([home_df, away_df]).sort_values(by=['team_id', 'date_utc']).reset_index(drop=True)
        
        # Calculate points
        def calc_pts(row):
            if row['gf'] > row['ga']: return 3
            if row['gf'] == row['ga']: return 1
            return 0
            
        team_matches['pts'] = team_matches.apply(calc_pts, axis=1)
        
        # Calculate rest days
        team_matches['prev_match_date'] = team_matches.groupby('team_id')['date_utc'].shift(1)
        team_matches['rest_days'] = (team_matches['date_utc'] - team_matches['prev_match_date']).dt.days
        team_matches['rest_days'] = team_matches['rest_days'].fillna(self.max_rest_days).clip(upper=self.max_rest_days)
        
        grouped = team_matches.groupby('team_id')
        
        for w in self.windows:
            team_matches[f'roll_pts_{w}'] = grouped['pts'].transform(lambda x: x.rolling(w, min_periods=min(w, 5)).mean().shift(1))
            team_matches[f'roll_gf_{w}'] = grouped['gf'].transform(lambda x: x.rolling(w, min_periods=min(w, 5)).mean().shift(1))
            team_matches[f'roll_ga_{w}'] = grouped['ga'].transform(lambda x: x.rolling(w, min_periods=min(w, 5)).mean().shift(1))
            team_matches[f'roll_xg_for_{w}'] = grouped['xg_for'].transform(lambda x: x.rolling(w, min_periods=min(w, 5)).mean().shift(1))
            team_matches[f'roll_xg_against_{w}'] = grouped['xg_against'].transform(lambda x: x.rolling(w, min_periods=min(w, 5)).mean().shift(1))
            team_matches[f'roll_xt_for_{w}'] = grouped['xt_for'].transform(lambda x: x.rolling(w, min_periods=min(w, 5)).mean().shift(1))
            team_matches[f'roll_poss_{w}'] = grouped['poss_for'].transform(lambda x: x.rolling(w, min_periods=min(w, 5)).mean().shift(1))
            team_matches[f'roll_sot_for_{w}'] = grouped['sot_for'].transform(lambda x: x.rolling(w, min_periods=min(w, 5)).mean().shift(1))
        
        # Merge back
        join_cols = ['match_row_id', 'team_id', 'rest_days'] + [c for c in team_matches.columns if c.startswith('roll_')]
        
        final_df = matches_df.copy()
        
        home_feats = team_matches[team_matches['is_home'] == 1][join_cols]
        home_feats.columns = ['id', 'home_team_id', 'home_rest_days'] + [f"home_{c}" for c in join_cols[3:]]
        
        away_feats = team_matches[team_matches['is_home'] == 0][join_cols]
        away_feats.columns = ['id', 'away_team_id', 'away_rest_days'] + [f"away_{c}" for c in join_cols[3:]]
        
        final_df = final_df.merge(home_feats, on=['id', 'home_team_id'], how='left')
        final_df = final_df.merge(away_feats, on=['id', 'away_team_id'], how='left')
        
        # Calculate deltas
        for w in self.windows:
            final_df[f'delta_pts_{w}'] = final_df[f'home_roll_pts_{w}'] - final_df[f'away_roll_pts_{w}']
            final_df[f'delta_xg_for_{w}'] = final_df[f'home_roll_xg_for_{w}'] - final_df[f'away_roll_xg_for_{w}']
            final_df[f'delta_xt_for_{w}'] = final_df[f'home_roll_xt_for_{w}'] - final_df[f'away_roll_xt_for_{w}']
            final_df[f'delta_poss_{w}'] = (final_df[f'home_roll_poss_{w}'] - final_df[f'away_roll_poss_{w}']).fillna(0)
            final_df[f'delta_sot_for_{w}'] = (final_df[f'home_roll_sot_for_{w}'] - final_df[f'away_roll_sot_for_{w}']).fillna(0)
            
        final_df['delta_rest_days'] = final_df['home_rest_days'] - final_df['away_rest_days']
        
        # 4. Referee Features & Market Odds
        # Calculation: Average cards per referee (historical)
        def total_cards(row):
            try:
                # API-Football cards come from events or stats. 
                # For historical data, we rely on what was ingested.
                hy = row.get('home_yellows', 0) or 0
                ay = row.get('away_yellows', 0) or 0
                return hy + ay
            except: return 0
            
        matches_df['total_cards_observed'] = matches_df.apply(total_cards, axis=1)
        
        # Compute Referee Rolling Averages
        ref_groups = matches_df.groupby('referee')
        matches_df['referee_cards_avg'] = ref_groups['total_cards_observed'].transform(lambda x: x.rolling(20, min_periods=1).mean().shift(1))
        matches_df['referee_cards_avg'] = matches_df['referee_cards_avg'].fillna(4.0)
        
        # Bring referee_cards_avg into final_df
        # final_df is a copy of matches_df, but it has undergone merges. 
        # To be safe, we merge only the newly calculated ref avg.
        final_df = final_df.merge(matches_df[['id', 'referee_cards_avg']], on='id', how='left')
        
        # Convert odds to implied probabilities (using columns already in final_df)
        def get_prob(odd):
            if pd.isna(odd) or odd <= 0: return 0.0
            return 1.0 / odd
            
        final_df['feat_prob_home'] = final_df['odds_1x2_home'].apply(get_prob)
        final_df['feat_prob_draw'] = final_df['odds_1x2_draw'].apply(get_prob)
        final_df['feat_prob_away'] = final_df['odds_1x2_away'].apply(get_prob)
        
        # Re-normalize to remove overround
        total_p = final_df['feat_prob_home'] + final_df['feat_prob_draw'] + final_df['feat_prob_away']
        mask = total_p > 0
        final_df.loc[mask, 'feat_prob_home'] /= total_p[mask]
        final_df.loc[mask, 'feat_prob_draw'] /= total_p[mask]
        final_df.loc[mask, 'feat_prob_away'] /= total_p[mask]
        
        return final_df
