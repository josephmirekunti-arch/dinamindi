import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss

class WalkForwardEvaluator:
    def __init__(self, model, train_window_days=365, test_window_days=30):
        self.model = model
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days

    def evaluate(self, df: pd.DataFrame, feature_cols: list[str]):
        """
        Runs walk-forward evaluation.
        df must be sorted by date_utc.
        """
        df = df.sort_values('date_utc').reset_index(drop=True)
        min_date = df['date_utc'].min()
        max_date = df['date_utc'].max()
        
        current_date = min_date + pd.Timedelta(days=self.train_window_days)
        
        results = []
        
        while current_date < max_date:
            train_mask = (df['date_utc'] < current_date)
            test_mask = (df['date_utc'] >= current_date) & (df['date_utc'] < current_date + pd.Timedelta(days=self.test_window_days))
            
            train_df = df[train_mask]
            test_df = df[test_mask]
            
            if len(test_df) == 0:
                current_date += pd.Timedelta(days=self.test_window_days)
                continue
                
            # Time decay weights for training
            # w_j = exp(-gamma * days_ago)
            # gamma = 0.02
            days_ago = (current_date - train_df['date_utc']).dt.days
            weights = np.exp(-0.02 * days_ago).values
            
            # Fit model
            self.model.fit(train_df, team_cols=[], covariate_cols=feature_cols, weights=weights)
            
            # Predict test set
            preds = self.model.predict_lambdas(test_df, feature_cols)
            
            # Combine
            fold_results = test_df[['id', 'date_utc', 'home_team_id', 'away_team_id', 'home_goals_ft', 'away_goals_ft']].copy()
            fold_results['lambda_h'] = preds['lambda_h'].values
            fold_results['lambda_a'] = preds['lambda_a'].values
            
            results.append(fold_results)
            
            current_date += pd.Timedelta(days=self.test_window_days)
            
        return pd.concat(results).reset_index(drop=True) if results else pd.DataFrame()

    def calc_metrics(self, results_df: pd.DataFrame, prob_deriver):
        """
        Applies probability derivation and calculates Log Loss and Brier Score.
        """
        probs = []
        for idx, row in results_df.iterrows():
            p = prob_deriver.derive_markets(row['lambda_h'], row['lambda_a'])
            probs.append(p)
            
        prob_df = pd.DataFrame(probs)
        results = pd.concat([results_df, prob_df], axis=1)
        
        # Calculate actual outcomes
        def get_1x2(row):
            if row['home_goals_ft'] > row['away_goals_ft']: return 'H'
            if row['home_goals_ft'] == row['away_goals_ft']: return 'D'
            return 'A'
            
        results['actual_1x2'] = results.apply(get_1x2, axis=1)
        results['actual_o25'] = (results['home_goals_ft'] + results['away_goals_ft']) > 2.5
        results['actual_btts'] = (results['home_goals_ft'] > 0) & (results['away_goals_ft'] > 0)
        
        # Log loss for 1X2
        # Need to construct y_true as one-hot and y_pred
        y_true_1x2 = pd.get_dummies(results['actual_1x2'])[['H', 'D', 'A']].values
        y_pred_1x2 = results[['P_H', 'P_D', 'P_A']].values
        ll_1x2 = log_loss(y_true_1x2, y_pred_1x2)
        
        # Ranked Probability Score (RPS) for 1X2
        # Ordinal outcome order: Home Win (0), Draw (1), Away Win (2)
        y_ord = results['actual_1x2'].map({'H': 0, 'D': 1, 'A': 2}).values
        P1 = results['P_H'].values
        P2 = results['P_H'].values + results['P_D'].values
        
        Y1 = (y_ord == 0).astype(int)
        Y2 = (y_ord <= 1).astype(int)
        
        # RPS Formula: 1/(r-1) * sum((P_i - Y_i)^2) where r=3
        rps_1x2 = 0.5 * np.mean((P1 - Y1)**2 + (P2 - Y2)**2)
        
        # Brier score for O2.5
        brier_o25 = brier_score_loss(results['actual_o25'], results['P_O25'])
        
        # Brier score for BTTS
        brier_btts = brier_score_loss(results['actual_btts'], results['P_BTTS_Y'])
        
        return {
            'log_loss_1x2': ll_1x2,
            'rps_1x2': rps_1x2,
            'brier_o25': brier_o25,
            'brier_btts': brier_btts
        }, results
