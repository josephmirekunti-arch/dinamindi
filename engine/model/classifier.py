import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import os

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from .rnn_model import GRUClassifierWrapper

class MatchClassifier:
    def __init__(self, n_estimators=150, max_depth=5, learning_rate=0.05):
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            subsample=0.8,
            colsample_bytree=0.8
        )
        
        if HAS_CATBOOST:
            self.cat_model = CatBoostClassifier(
                iterations=n_estimators,
                depth=max_depth,
                learning_rate=learning_rate,
                loss_function='MultiClass',
                verbose=False,
                random_state=42
            )
        else:
            self.cat_model = None
            
        self.gru_model = GRUClassifierWrapper()
        
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
        Trains the XGBoost, CatBoost, and GRU models on historical match data.
        """
        if train_df.empty:
            return

        X = train_df[feature_cols].copy()
        # Fill NaNs
        X = X.fillna(0)
        
        y = self._prepare_target(train_df)
        
        self.xgb_model.fit(X, y)
        if self.cat_model:
            self.cat_model.fit(X, y)
            
        self.gru_model.fit(train_df, y)
        
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
        
        probs_xgb = self.xgb_model.predict_proba(X_feat)
        
        if self.cat_model:
            probs_cat = self.cat_model.predict_proba(X_feat)
        else:
            probs_cat = probs_xgb
            
        probs_gru = self.gru_model.predict_proba(X)
        
        # Ensemble Average
        # Weighting: 40% XGBoost, 40% CatBoost, 20% GRU (if GRU is trained)
        if hasattr(self.gru_model, 'is_trained') and self.gru_model.is_trained:
            probs = (probs_xgb * 0.4) + (probs_cat * 0.4) + (probs_gru * 0.2)
        else:
            probs = (probs_xgb * 0.5) + (probs_cat * 0.5)
        
        # Determine mapping from label encoder
        # Typically ['A', 'D', 'H'] -> 0, 1, 2
        classes = self.label_encoder.classes_
        prob_dict = {}
        for i, label in enumerate(classes):
            if label == 'H': prob_dict['prob_h'] = probs[:, i]
            if label == 'D': prob_dict['prob_d'] = probs[:, i]
            if label == 'A': prob_dict['prob_a'] = probs[:, i]
            
        return pd.DataFrame(prob_dict, index=X.index)
