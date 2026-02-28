import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import statsmodels.api as sm

class PoissonGoalsModel:
    def __init__(self, alpha=1.0):
        # We use Ridge regression on log-transformed expected goals (approx for Poisson)
        # For a true GLM with L2 penalty, statsmodels has GLM fit_regularized
        self.alpha = alpha
        self.model = None
        self.coefs = {}
        self.intercept = 0.0

    def fit(self, features_df: pd.DataFrame, team_cols: list[str], covariate_cols: list[str], weights: np.ndarray = None):
        """
        Fits a separate GLM for goals using a Poisson link function, with 
        dummy variables for team attack/defense str.
        
        Since setting up Ridge on Poisson via statsmodels can be slow/brittle sometimes,
        we can approximate via regularized OLS on log(goals + 1) or stick to scikit-learn Ridge.
        For true Poisson, statsmodels GLM is standard. Here we'll configure statsmodels GLM.
        """
        # We need to construct a dataset where each row is a team's goals in a match (home and away separated).
        
        # 1. Build Home rows
        home_y = features_df['home_goals_ft']
        # 2. Build Away rows
        away_y = features_df['away_goals_ft']
        
        # This will be refined to fully implement the hierarchical attack/defense model.
        # Let's start with a simpler Scikit-Learn PoissonRegressor if available, 
        # but Scikit-Learn expects >=1.2 for PoissonRegressor. requirements.txt has 1.4.0, so yes!
        
        from sklearn.linear_model import PoissonRegressor
        
        self.model_h = PoissonRegressor(alpha=self.alpha, max_iter=1000)
        self.model_a = PoissonRegressor(alpha=self.alpha, max_iter=1000)
        
        home_X = features_df[covariate_cols] 
        away_X = features_df[covariate_cols] # typically you negate deltas for the away perspective, or use dedicated away cols
        
        # Fit models
        self.model_h.fit(home_X, home_y, sample_weight=weights)
        self.model_a.fit(away_X, away_y, sample_weight=weights)
        
    def predict_lambdas(self, X: pd.DataFrame, covariate_cols: list[str]) -> pd.DataFrame:
        """
        Predicts lambda_h and lambda_a for a set of fixtures.
        """
        lam_h = self.model_h.predict(X[covariate_cols])
        # Assume X for away model needs identical structure but inverted deltas if we used deltas
        # Adjust based on how features are shaped!
        lam_a = self.model_a.predict(X[covariate_cols])
        
        return pd.DataFrame({
            'lambda_h': lam_h,
            'lambda_a': lam_a
        })
