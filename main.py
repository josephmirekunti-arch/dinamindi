import yaml
import pandas as pd
from engine.store.db import DbStore, Match
from engine.ingest.understat import UnderstatIngestor
from engine.validate.schema import MatchValidator
from engine.features.rolling import FeatureEngineer
from engine.model.poisson import PoissonGoalsModel
from engine.predict.probs import ProbabilityDeriver
from engine.backtest.walk_forward import WalkForwardEvaluator

def main():
    print("Loading config...")
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    print("Initializing Database...")
    store = DbStore()
    store.create_all()
    session = store.get_session()
    
    # 1. Ingestion Phase
    target_season = '2025'
    competitions = config.get('competitions_in_scope', ['EPL', 'La_liga', 'Bundesliga'])
    print(f"Ingesting {competitions} for {target_season} Season...")
    ingestor = UnderstatIngestor()
    matches = []
    for comp in competitions:
        try:
            comp_matches = ingestor.fetch_season_matches(comp, target_season)
            matches.extend(comp_matches)
        except Exception as e:
            print(f"Failed to fetch Understat data for {comp}: {e}")
    
    # 2. Validation Phase
    validator = MatchValidator()
    # We want to keep unplayed matches for prediction, so modifying validation locally here
    # to not drop unplayed matches. The validator in schema.py drops unplayed right now if goals are None?
    # Actually validation drops if is_played and goals are none. Unplayed is fine!
    valid_matches = validator.validate_batch(matches)
    print(f"Validated {len(valid_matches)} matches.")
    
    df_data = [{
        'id': m.match_id,
        'date_utc': m.date_utc,
        'competition': m.competition,
        'season': m.season,
        'home_team_id': m.home_team_id,
        'away_team_id': m.away_team_id,
        'home_team_name': m.home_team_name,
        'away_team_name': m.away_team_name,
        'home_goals_ft': m.home_goals_ft,
        'away_goals_ft': m.away_goals_ft,
        'home_xg': m.home_xg,
        'away_xg': m.away_xg,
        'is_played': m.is_played
    } for m in valid_matches]
    
    matches_df = pd.DataFrame(df_data)
    
    # 3. Feature Engineering Phase
    print("Engineering features...")
    fe = FeatureEngineer(
        windows=config.get('rolling_windows', [5, 10, 20]),
        max_rest_days=14
    )
    features_df = fe.compute_features(matches_df)
    
    # Select features generated
    feature_cols = [c for c in features_df.columns if c.startswith('delta_')]
    
    # Drop rows with NaNs caused by the rolling window on early matches
    # Specifically check only the feature columns, because future matches have NaN for goals/xG!
    features_df = features_df.dropna(subset=feature_cols)
    print(f"Features created. {len(features_df)} matches available post rolling window.")
    
    # Split into Played (Training) and Unplayed (Predicting)
    train_df = features_df[features_df['is_played'] == True].copy()
    predict_df = features_df[features_df['is_played'] == False].copy()
    
    print(f"Training on {len(train_df)} past matches...")
    print(f"Generating predictions for {len(predict_df)} upcoming fixtures...")
    
    if len(train_df) < 50:
        print("Not enough played matches found to run reliable model.")
        return
        
    # 4. Modeling Phase
    model = PoissonGoalsModel(alpha=1.0) # Ridge penalty
    feature_cols = [c for c in train_df.columns if c.startswith('delta_')]
    
    weights = np.ones(len(train_df))
    model.fit(train_df, team_cols=[], covariate_cols=feature_cols, weights=weights)
    print("Model trained on ongoing season.")
    
    if len(predict_df) == 0:
        print("No matches left to predict in the current dataset.")
        return
        
    # 5. Prediction Phase
    print("\n--- UPCOMING FIXTURE PREDICTIONS ---")
    preds = model.predict_lambdas(predict_df, feature_cols)
    deriver = ProbabilityDeriver(max_goals=config.get('max_goals', 8))
    
    # Store predictions back into dataframe for visualization
    predict_df['lambda_h'] = preds['lambda_h'].values
    predict_df['lambda_a'] = preds['lambda_a'].values
    
    predict_df = predict_df.sort_values(by='date_utc')
    thresholds = config.get('selection_thresholds', {})
    
    for idx, row in predict_df.head(10).iterrows():
        mkt = deriver.derive_markets(row['lambda_h'], row['lambda_a'])
        recs = deriver.get_recommendations(mkt, thresholds)
        
        home = row['home_team_name']
        away = row['away_team_name']
        date = row['date_utc'].strftime("%Y-%m-%d %H:%M")
        
        print(f"[{date}] {home} vs {away}")
        print(f"  Exp. Goals: {home} {row['lambda_h']:.2f} - {row['lambda_a']:.2f} {away}")
        print(f"  1X2 Probs: H: {mkt['P_H']:.1%} | D: {mkt['P_D']:.1%} | A: {mkt['P_A']:.1%}")
        print(f"  O2.5 Prob: {mkt['P_O25']:.1%} | BTTS Prob: {mkt['P_BTTS_Y']:.1%}")
        
        if recs:
            print(f"  *** RECOMMENDED STAKES *** -> {', '.join(recs)}")
        else:
            print(f"  -> No markets pass the integrity threshold. Skip staking.")
            
        print("-" * 50)

if __name__ == "__main__":
    import numpy as np
    main()
