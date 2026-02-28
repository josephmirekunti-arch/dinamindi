import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, windows=[5, 10, 20], max_rest_days=14):
        self.windows = windows
        self.max_rest_days = max_rest_days

    def _compute_elo(self, matches_df: pd.DataFrame) -> pd.DataFrame:
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
            
            # Season reversion
            if team_last_season.get(h_id, season) != season:
                r_h = r_h * 0.8 + 1500.0 * 0.2
            if team_last_season.get(a_id, season) != season:
                r_a = r_a * 0.8 + 1500.0 * 0.2
                
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

    def compute_features(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a DataFrame of historical matches, chronologically sorted, 
        and computes rolling window features for home and away teams.
        This must be strictly pre-match (shift by 1).
        """
        # Compute Elo sequentially
        matches_df = self._compute_elo(matches_df)
        
        # Ensure we are sorted by date
        matches_df = matches_df.sort_values(by='date_utc').reset_index(drop=True)
        
        # Melt the dataframe so each row is a team-match with context
        home_cols = ['id', 'date_utc', 'competition', 'season', 'home_team_id', 'home_goals_ft', 'away_goals_ft', 'home_xg', 'away_xg', 'home_possession', 'away_possession', 'home_sot', 'away_sot', 'home_corners', 'away_corners', 'home_yellows', 'away_yellows', 'home_reds', 'away_reds']
        away_cols = ['id', 'date_utc', 'competition', 'season', 'away_team_id', 'away_goals_ft', 'home_goals_ft', 'away_xg', 'home_xg', 'away_possession', 'home_possession', 'away_sot', 'home_sot', 'away_corners', 'home_corners', 'away_yellows', 'home_yellows', 'away_reds', 'home_reds']
        
        # We fill API-Football specific stats with NaNs initially. 
        # If the API key is not present, we want these omitted or zeroed in modeling without breaking.
        for col in home_cols + away_cols:
            if col not in matches_df.columns:
                matches_df[col] = np.nan
        
        home_df = matches_df[home_cols].copy()
        home_df['is_home'] = 1
        home_df.columns = ['match_row_id', 'date_utc', 'competition', 'season', 'team_id', 'gf', 'ga', 'xg_for', 'xg_against', 'poss_for', 'poss_against', 'sot_for', 'sot_against', 'corners_for', 'corners_against', 'yellows_for', 'yellows_against', 'reds_for', 'reds_against', 'is_home']
        
        away_df = matches_df[away_cols].copy()
        away_df['is_home'] = 0
        away_df.columns = ['match_row_id', 'date_utc', 'competition', 'season', 'team_id', 'gf', 'ga', 'xg_for', 'xg_against', 'poss_for', 'poss_against', 'sot_for', 'sot_against', 'corners_for', 'corners_against', 'yellows_for', 'yellows_against', 'reds_for', 'reds_against', 'is_home']
        
        team_matches = pd.concat([home_df, away_df]).sort_values(by=['team_id', 'date_utc']).reset_index(drop=True)
        
        # Calculate points
        def calc_pts(row):
            if row['gf'] > row['ga']: return 3
            if row['gf'] == row['ga']: return 1
            return 0
            
        team_matches['pts'] = team_matches.apply(calc_pts, axis=1)
        team_matches['gd'] = team_matches['gf'] - team_matches['ga']
        
        # Calculate rest days
        team_matches['prev_match_date'] = team_matches.groupby('team_id')['date_utc'].shift(1)
        team_matches['rest_days'] = (team_matches['date_utc'] - team_matches['prev_match_date']).dt.days
        team_matches['rest_days'] = team_matches['rest_days'].fillna(self.max_rest_days).clip(upper=self.max_rest_days)
        
        features = []
        grouped = team_matches.groupby('team_id')
        
        for w in self.windows:
            team_matches[f'roll_pts_{w}'] = grouped['pts'].transform(lambda x: x.rolling(w, min_periods=min(w, 5)).mean().shift(1))
            team_matches[f'roll_gf_{w}'] = grouped['gf'].transform(lambda x: x.rolling(w, min_periods=min(w, 5)).mean().shift(1))
            team_matches[f'roll_ga_{w}'] = grouped['ga'].transform(lambda x: x.rolling(w, min_periods=min(w, 5)).mean().shift(1))
            team_matches[f'roll_xg_for_{w}'] = grouped['xg_for'].transform(lambda x: x.rolling(w, min_periods=min(w, 5)).mean().shift(1))
            team_matches[f'roll_xg_against_{w}'] = grouped['xg_against'].transform(lambda x: x.rolling(w, min_periods=min(w, 5)).mean().shift(1))
            
            # API Football specific
            team_matches[f'roll_poss_{w}'] = grouped['poss_for'].transform(lambda x: x.rolling(w, min_periods=min(w, 5)).mean().shift(1))
            team_matches[f'roll_sot_for_{w}'] = grouped['sot_for'].transform(lambda x: x.rolling(w, min_periods=min(w, 5)).mean().shift(1))
            
            # Secondary Markets
            team_matches[f'roll_corners_for_{w}'] = grouped['corners_for'].transform(lambda x: x.rolling(w, min_periods=min(w, 5)).mean().shift(1))
            team_matches[f'roll_corners_against_{w}'] = grouped['corners_against'].transform(lambda x: x.rolling(w, min_periods=min(w, 5)).mean().shift(1))
            team_matches[f'roll_cards_for_{w}'] = grouped['yellows_for'].transform(lambda x: x.rolling(w, min_periods=min(w, 5)).mean().shift(1))
            team_matches[f'roll_cards_against_{w}'] = grouped['yellows_against'].transform(lambda x: x.rolling(w, min_periods=min(w, 5)).mean().shift(1))
        
        # Now we need to merge these team-level pre-match features back to the main matches_df
        join_cols = ['match_row_id', 'team_id', 'rest_days'] + [c for c in team_matches.columns if c.startswith('roll_')]
        
        final_df = matches_df.copy()
        
        home_feats = team_matches[team_matches['is_home'] == 1][join_cols]
        home_feats.columns = ['id', 'home_team_id', 'home_rest_days'] + [f"home_{c}" for c in join_cols[3:]]
        
        away_feats = team_matches[team_matches['is_home'] == 0][join_cols]
        away_feats.columns = ['id', 'away_team_id', 'away_rest_days'] + [f"away_{c}" for c in join_cols[3:]]
        
        final_df = final_df.merge(home_feats, on=['id', 'home_team_id'], how='left')
        final_df = final_df.merge(away_feats, on=['id', 'away_team_id'], how='left')
        
        # Calculate deltas for convenience in modelling
        for w in self.windows:
            final_df[f'delta_pts_{w}'] = final_df[f'home_roll_pts_{w}'] - final_df[f'away_roll_pts_{w}']
            final_df[f'delta_gf_{w}'] = final_df[f'home_roll_gf_{w}'] - final_df[f'away_roll_gf_{w}']
            final_df[f'delta_ga_{w}'] = final_df[f'home_roll_ga_{w}'] - final_df[f'away_roll_ga_{w}']
            
            # Understat
            final_df[f'delta_xg_for_{w}'] = final_df[f'home_roll_xg_for_{w}'] - final_df[f'away_roll_xg_for_{w}']
            final_df[f'delta_xg_against_{w}'] = final_df[f'home_roll_xg_against_{w}'] - final_df[f'away_roll_xg_against_{w}']
            
            # API-Football
            # We fillna(0) so the regression model doesn't drop columns entirely if predicting without API footy data
            final_df[f'delta_poss_{w}'] = (final_df[f'home_roll_poss_{w}'] - final_df[f'away_roll_poss_{w}']).fillna(0)
            final_df[f'delta_sot_for_{w}'] = (final_df[f'home_roll_sot_for_{w}'] - final_df[f'away_roll_sot_for_{w}']).fillna(0)
            
            # Secondary Stats
            final_df[f'delta_corners_{w}'] = (final_df[f'home_roll_corners_for_{w}'] - final_df[f'away_roll_corners_for_{w}']).fillna(0)
            final_df[f'delta_cards_{w}'] = (final_df[f'home_roll_cards_for_{w}'] - final_df[f'away_roll_cards_for_{w}']).fillna(0)
            
        final_df['delta_rest_days'] = final_df['home_rest_days'] - final_df['away_rest_days']
        
        return final_df
