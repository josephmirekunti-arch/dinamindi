import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import os

class MatchClassifier:
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='multi:softprob',
            num_class=3,
            random_state=42
        )
        self.label_encoder = LabelEncoder()
        # Mapping: 0: Away, 1: Draw, 2: Home (Standard for sorted ['A', 'D', 'H'])
        self.is_trained = False

    def _prepare_target(self, df: pd.DataFrame) -> np.ndarray:
        def get_result(row):
            if row['home_goals_ft'] > row['away_goals_ft']: return 'H'
            if row['home_goals_ft'] < row['away_goals_ft']: return 'A'
            return 'D'
        
        results = df.apply(get_result, axis=1)
        return self.label_encoder.fit_transform(results)

    def fit(self, train_df: pd.DataFrame, feature_cols: list[str]):
        """
        Trains the XGBoost model on historical match data.
        """
        if train_df.empty:
            return

        X = train_df[feature_cols].copy()
        # Fill NaNs for XGBoost (though it handles them, we want consistency)
        X = X.fillna(0)
        
        y = self._prepare_target(train_df)
        
        self.model.fit(X, y)
        self.is_trained = True

    def predict_probs(self, X: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """
        Predicts 1X2 probabilities.
        Returns a DataFrame with columns ['prob_h', 'prob_d', 'prob_a']
        """
        if not self.is_trained:
            # Return uniform prior if not trained
            return pd.DataFrame({
                'prob_h': [0.33] * len(X),
                'prob_d': [0.34] * len(X),
                'prob_a': [0.33] * len(X)
            }, index=X.index)

        X_feat = X[feature_cols].copy().fillna(0)
        probs = self.model.predict_proba(X_feat)
        
        # Determine mapping from label encoder
        # Typically ['A', 'D', 'H'] -> 0, 1, 2
        classes = self.label_encoder.classes_
        prob_dict = {}
        for i, label in enumerate(classes):
            if label == 'H': prob_dict['prob_h'] = probs[:, i]
            if label == 'D': prob_dict['prob_d'] = probs[:, i]
            if label == 'A': prob_dict['prob_a'] = probs[:, i]
            
        return pd.DataFrame(prob_dict, index=X.index)
