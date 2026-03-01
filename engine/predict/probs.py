import numpy as np
from scipy.stats import poisson

class ProbabilityDeriver:
    def __init__(self, max_goals=8):
        self.max_goals = max_goals
        
    def _goals_matrix(self, lambda_h, lambda_a):
        """
        Computes the max_goals x max_goals matrix of probabilities P(h=i, a=j).
        """
        h_prob = [poisson.pmf(i, lambda_h) for i in range(self.max_goals + 1)]
        a_prob = [poisson.pmf(j, lambda_a) for j in range(self.max_goals + 1)]
        return np.outer(h_prob, a_prob)
        
    def derive_markets(self, lambda_h: float, lambda_a: float, classifier_probs: dict = None, weight_poisson=0.5) -> dict:
        """
        Derives P(H), P(D), P(A), O/U 0.5, 1.5, 2.5, BTTS, DNB, and AH.
        If classifier_probs is provided, 1X2 outcomes are blended.
        """
        mat = self._goals_matrix(lambda_h, lambda_a)
        
        # Poisson-derived 1X2
        p_h_poi = np.sum(np.tril(mat, -1))
        p_d_poi = np.sum(np.diag(mat))
        p_a_poi = np.sum(np.triu(mat, 1))
        
        # Ensemble blending for 1X2
        if classifier_probs:
            # weight_poisson=0.5 means equal voting as per simple ensemble best practices
            weight_ml = 1.0 - weight_poisson
            p_h = (p_h_poi * weight_poisson) + (classifier_probs['prob_h'] * weight_ml)
            p_d = (p_d_poi * weight_poisson) + (classifier_probs['prob_d'] * weight_ml)
            p_a = (p_a_poi * weight_poisson) + (classifier_probs['prob_a'] * weight_ml)
            
            # Re-normalize to ensure sum is exactly 1.0
            total = p_h + p_d + p_a
            p_h /= total
            p_d /= total
            p_a /= total
        else:
            p_h, p_d, p_a = p_h_poi, p_d_poi, p_a_poi

        # Over/Under 0.5, 1.5, 2.5
        p_u05 = mat[0, 0]
        p_o05 = 1.0 - p_u05
        
        p_u15 = mat[0, 0] + mat[1, 0] + mat[0, 1]
        p_o15 = 1.0 - p_u15
        
        p_u25 = 0.0
        for i in range(3):
            for j in range(3-i):
                p_u25 += mat[i, j]
        p_o25 = 1.0 - p_u25
        
        # BTTS
        p_btts_n = sum(mat[i, 0] for i in range(self.max_goals + 1)) + \
                   sum(mat[0, j] for j in range(1, self.max_goals + 1)) - mat[0, 0]
        p_btts_y = 1.0 - p_btts_n
        
        # Draw No Bet (DNB)
        # Normalized by removing the draw probability
        p_dnb_h = p_h / (p_h + p_a) if (p_h + p_a) > 0 else 0.0
        p_dnb_a = p_a / (p_h + p_a) if (p_h + p_a) > 0 else 0.0
        
        # Asian Handicap -1.5 / + 1.5
        # Home AH -1.5 means Home must win by 2 or more
        p_ah_h_minus_15 = np.sum(np.tril(mat, -2))
        p_ah_a_plus_15 = 1.0 - p_ah_h_minus_15
        
        # Away AH -1.5 means Away must win by 2 or more
        p_ah_a_minus_15 = np.sum(np.triu(mat, 2))
        p_ah_h_plus_15 = 1.0 - p_ah_a_minus_15
        
        return {
            'P_H': float(p_h),
            'P_D': float(p_d),
            'P_A': float(p_a),
            'P_O05': float(p_o05),
            'P_U05': float(p_u05),
            'P_O15': float(p_o15),
            'P_U15': float(p_u15),
            'P_O25': float(p_o25),
            'P_U25': float(p_u25),
            'P_BTTS_Y': float(p_btts_y),
            'P_BTTS_N': float(p_btts_n),
            'P_DNB_H': float(p_dnb_h),
            'P_DNB_A': float(p_dnb_a),
            'P_AH_H_m15': float(p_ah_h_minus_15),
            'P_AH_A_p15': float(p_ah_a_plus_15),
            'P_AH_A_m15': float(p_ah_a_minus_15),
            'P_AH_H_p15': float(p_ah_h_plus_15)
        }

    def get_recommendations(self, mkt: dict, thresholds: dict) -> list[str]:
        """
        Evaluates the market probabilities against configured thresholds 
        and returns a list of recommended bets for all markets.
        """
        recommendations = []
        
        # Binary markets
        bin_thresh = thresholds.get('binary_markets', 0.55)
        
        # Over/Under
        if mkt['P_O05'] >= bin_thresh * 1.5:  # Scaled up because O0.5 is almost always > 80%
            recommendations.append(f"OVER 0.5 ({mkt['P_O05']:.1%})")
        if mkt['P_U05'] >= bin_thresh:
            recommendations.append(f"UNDER 0.5 ({mkt['P_U05']:.1%})")
            
        if mkt['P_O15'] >= bin_thresh * 1.2:
            recommendations.append(f"OVER 1.5 ({mkt['P_O15']:.1%})")
        elif mkt['P_U15'] >= bin_thresh:
            recommendations.append(f"UNDER 1.5 ({mkt['P_U15']:.1%})")
            
        if mkt['P_O25'] >= bin_thresh:
            recommendations.append(f"OVER 2.5 ({mkt['P_O25']:.1%})")
        elif mkt['P_U25'] >= bin_thresh:
            recommendations.append(f"UNDER 2.5 ({mkt['P_U25']:.1%})")
            
        # BTTS
        if mkt['P_BTTS_Y'] >= bin_thresh:
            recommendations.append(f"BTTS YES ({mkt['P_BTTS_Y']:.1%})")
        elif mkt['P_BTTS_N'] >= bin_thresh:
            recommendations.append(f"BTTS NO ({mkt['P_BTTS_N']:.1%})")
            
        # Draw No Bet
        if mkt['P_DNB_H'] >= bin_thresh:
            recommendations.append(f"DNB HOME ({mkt['P_DNB_H']:.1%})")
        elif mkt['P_DNB_A'] >= bin_thresh:
            recommendations.append(f"DNB AWAY ({mkt['P_DNB_A']:.1%})")
            
        # Asian Handicap
        if mkt['P_AH_H_m15'] >= bin_thresh:
            recommendations.append(f"AH HOME -1.5 ({mkt['P_AH_H_m15']:.1%})")
        if mkt['P_AH_A_p15'] >= bin_thresh:
            recommendations.append(f"AH AWAY +1.5 ({mkt['P_AH_A_p15']:.1%})")
        if mkt['P_AH_A_m15'] >= bin_thresh:
            recommendations.append(f"AH AWAY -1.5 ({mkt['P_AH_A_m15']:.1%})")
        if mkt['P_AH_H_p15'] >= bin_thresh:
            recommendations.append(f"AH HOME +1.5 ({mkt['P_AH_H_p15']:.1%})")
            
        # 1X2 market
        top_prob_thresh = thresholds.get('1x2_top_prob', 0.50)
        margin_thresh = thresholds.get('1x2_margin', 0.10)
        
        probs_1x2 = [
            ('HOME WINS', mkt['P_H']),
            ('DRAW', mkt['P_D']),
            ('AWAY WINS', mkt['P_A'])
        ]
        
        # Sort descending by probability
        probs_1x2.sort(key=lambda x: x[1], reverse=True)
        top_outcome, top_prob = probs_1x2[0]
        second_prob = probs_1x2[1][1]
        
        # Require absolute minimum probability AND a safe margin over the second most likely outcome
        if top_prob >= top_prob_thresh and (top_prob - second_prob) >= margin_thresh:
            recommendations.append(f"1X2: {top_outcome} ({top_prob:.1%})")
            
        return recommendations

    def get_structured_recommendations(self, mkt: dict, thresholds: dict, lam_h: float, lam_a: float, odds: dict = None) -> list[dict]:
        """
        Returns structured dicts with risk level, analysis, and EV logic for the UI.
        """
        if odds is None:
            odds = {}
            
        markets = []
        bin_thresh = thresholds.get('binary_markets', 0.55)
        
        def assess(prob, thresh_multiplier=1.0, market_name=""):
            threshold = bin_thresh * thresh_multiplier
            if prob >= threshold:
                risk = "recommended" # Green
                # Basic analysis templates
                if "OVER" in market_name or "YES" in market_name:
                    analysis = f"High expected goal volume ({lam_h:.1f} vs {lam_a:.1f}) drives strong {market_name} value."
                elif "UNDER" in market_name or "NO" in market_name:
                    analysis = f"Low expected event frequency provides solid foundation for {market_name}."
                elif "AH" in market_name:
                    analysis = f"Mathematical mismatch justifies clearing the handicap line."
                elif "HOME" in market_name:
                    analysis = f"Home form and mathematical expectation heavily favors the outcome."
                elif "AWAY" in market_name:
                    analysis = f"Away form sufficiently overcomes baseline home advantage."
                else:
                    analysis = "Statistically robust expected value."
            else:
                risk = "volatile" # Red
                analysis = f"High variability. Edge below {threshold:.1%} required threshold."
                
            # --- VALUE BETTING (+EV) CALCULATION ---
            bookie_odd = odds.get(market_name, None)
            implied_prob = None
            ev_status = "Unknown"
            ev_margin = 0.0
            
            if bookie_odd and bookie_odd > 0:
                implied_prob = 1.0 / bookie_odd
                # Expected Value = (Probability of Winning * Win Payout) - Probability of Losing * 1
                # Simplified: (Prob * Decimal Odd) - 1
                calc_ev = (prob * bookie_odd) - 1.0
                ev_margin = calc_ev
                
                if calc_ev > 0.05: # >5% edge
                    ev_status = "High Value (+EV)"
                elif calc_ev > 0:
                    ev_status = "Marginal Value (+EV)"
                else:
                    ev_status = "Avoid (-EV)"
                    
            return {
                "market": market_name,
                "probability": float(prob),
                "risk": risk,
                "analysis": analysis,
                "bookie_odd": bookie_odd,
                "implied_prob": float(implied_prob) if implied_prob else None,
                "ev_status": ev_status,
                "ev_margin": float(ev_margin)
            }
            
        # Add core markets to display
        markets.append(assess(mkt['P_O05'], 1.2, "OVER 0.5")) # Scale threshold up since O0.5 is almost always high
        markets.append(assess(mkt['P_U05'], 1.0, "UNDER 0.5"))
        markets.append(assess(mkt['P_O15'], 1.1, "OVER 1.5")) # Scale threshold slightly
        markets.append(assess(mkt['P_U15'], 1.0, "UNDER 1.5"))
        markets.append(assess(mkt['P_O25'], 1.0, "OVER 2.5"))
        markets.append(assess(mkt['P_U25'], 1.0, "UNDER 2.5"))
        markets.append(assess(mkt['P_BTTS_Y'], 1.0, "BTTS YES"))
        markets.append(assess(mkt['P_BTTS_N'], 1.0, "BTTS NO"))
        
        # 1X2 market structured
        top_prob_thresh = thresholds.get('1x2_top_prob', 0.50)
        probs_1x2 = [
            ('HOME WINS', mkt['P_H']),
            ('DRAW', mkt['P_D']),
            ('AWAY WINS', mkt['P_A'])
        ]
        probs_1x2.sort(key=lambda x: x[1], reverse=True)
        top_outcome, top_prob = probs_1x2[0]
        
        # Evaluate top 1X2 with the assess function to get EV
        market_str = f"1X2 {top_outcome}"
        market_ev = assess(top_prob, 1.0, market_str)
        
        # Override the static risk/analysis from assess if it doesn't meet 1x2 thresholds
        if top_prob < top_prob_thresh:
            market_ev["risk"] = "volatile"
            market_ev["analysis"] = "Uncertain match outcome. 1X2 market is highly volatile."
        else:
            market_ev["risk"] = "recommended"
            market_ev["analysis"] = "Strong win probability based on team expected goals differential."
            
        markets.append(market_ev)
        
        return markets

    def derive_elo_insights(self, home_elo: float, away_elo: float, lam_h: float, lam_a: float) -> dict:
        """
        Translates Elo differentials into specific market insights.
        """
        diff = home_elo - away_elo
        
        # Expected Win % based purely on historical Elo (including Home Advantage +80)
        implied_e_h = 1.0 / (1.0 + 10.0 ** ((away_elo - (home_elo + 80.0)) / 400.0))
        
        insight = {
            "verdict": "",
            "analysis": "",
            "recommended_markets": []
        }
        
        if diff >= 150:
            insight["verdict"] = "Heavy Home Favorite"
            insight["analysis"] = "Massive historical quality gap. Expect the home team to heavily control the game."
            insight["recommended_markets"] = ["AH Home -1.5", "Home Win to Nil"]
            if lam_h < 1.5:
                insight["analysis"] += " HOWEVER, recent home xG is surprisingly low. Potential trap, consider Under markets instead."
                insight["recommended_markets"] = ["Under 2.5", "Draw No Bet Home"]
        elif diff <= -150:
            insight["verdict"] = "Heavy Away Favorite"
            insight["analysis"] = "Away team vastly superior on paper. Home advantage unlikely to compensate."
            insight["recommended_markets"] = ["AH Away -1.0", "Away Over 1.5 Team Goals"]
            if lam_a < 1.2:
                insight["analysis"] += " Flag: Away team's recent offensive metrics are poor despite high Elo. Proceed with caution."
        elif -40 <= diff <= 40:
            insight["verdict"] = "Dead Heat"
            insight["analysis"] = "Teams are fundamentally equal in long-term quality. Match likely decided by tactical edge."
            insight["recommended_markets"] = ["Draw", "Under 2.5", "BTTS YES"]
        elif diff > 40:
            insight["verdict"] = "Home Advantage Lean"
            insight["analysis"] = "Home side is marginally superior. With home-field advantage, they hold the edge."
            insight["recommended_markets"] = ["Home WINS", "DNB Home"]
        else:
            insight["verdict"] = "Slight Away Lean"
            insight["analysis"] = "Away team is better, but home advantage neutralizes most of their edge. Very tight contest."
            insight["recommended_markets"] = ["DNB Away", "Double Chance X2"]
            
        return insight

    def derive_secondary_markets(self, lam_corners: float, lam_cards: float, threshold: float = 0.55) -> list[dict]:
        """
        Derives betting recommendations for Secondary Markets: Corners and Cards.
        Operates on total expected match values (Home expected + Away expected).
        """
        markets = []
        
        # Guard against zero/NaN from historical matches without API data
        if lam_corners == 0 or lam_cards == 0 or np.isnan(lam_corners) or np.isnan(lam_cards):
            return markets

        def assess(prob, market_name):
            if prob >= threshold:
                return {
                    "market": market_name,
                    "probability": float(prob),
                    "risk": "recommended",
                    "analysis": "Model expects high frequency based on recent team dynamics.",
                    "status": "UNKNOWN" # Will be updated in performance checker if we track it
                }
            return None

        # Corners Math (Poisson CDF)
        # Prob of exactly k corners
        # P(Under 8.5) = sum P(k) for k=0 to 8
        prob_u85_c = sum([poisson.pmf(k, lam_corners) for k in range(9)])
        prob_o85_c = 1.0 - prob_u85_c
        
        prob_u95_c = sum([poisson.pmf(k, lam_corners) for k in range(10)])
        prob_o95_c = 1.0 - prob_u95_c

        # Cards Math
        prob_u35_cards = sum([poisson.pmf(k, lam_cards) for k in range(4)])
        prob_o35_cards = 1.0 - prob_u35_cards
        
        prob_u45_cards = sum([poisson.pmf(k, lam_cards) for k in range(5)])
        prob_o45_cards = 1.0 - prob_u45_cards

        # Evaluate and append
        for p, name in [
            (prob_o85_c, "OVER 8.5 CORNERS"),
            (prob_u85_c, "UNDER 8.5 CORNERS"),
            (prob_o95_c, "OVER 9.5 CORNERS"),
            (prob_u95_c, "UNDER 9.5 CORNERS"),
            (prob_o35_cards, "OVER 3.5 CARDS"),
            (prob_u35_cards, "UNDER 3.5 CARDS"),
            (prob_o45_cards, "OVER 4.5 CARDS"),
            (prob_u45_cards, "UNDER 4.5 CARDS")
        ]:
            rec = assess(p, name)
            if rec: markets.append(rec)
            
        return markets

    def get_interval_stats(self, home_history: pd.DataFrame, away_history: pd.DataFrame):
        """
        Calculates goal scoring/conceding frequencies in 10 and 15 minute intervals.
        """
        import json
        
        def calculate_buckets(history, team_type='home'):
            # intervals: 10m (0-10, 11-20, ... 81-90+)
            # 15m (0-15, 16-30, ... 76-90+)
            buckets_10 = {i: 0 for i in range(0, 10)} # 0-9 inclusive for 10 buckets
            buckets_15 = {i: 0 for i in range(0, 6)}
            
            total_matches = len(history)
            if total_matches == 0:
                return {"10m": [0]*10, "15m": [0]*6}
            
            for _, row in history.iterrows():
                ev_str = row.get('goal_events')
                if not ev_str: continue
                try:
                    evs = json.loads(ev_str)
                    for ev in evs:
                        # Check if it's the team's goal or conceded
                        # In history, 'is_home' tells us if the team was home or away in that row
                        # but goal_events 'team' is 'home'/'away' relative to that match
                        # We need to know if this team scored it.
                        minute = min(int(ev['minute']), 90)
                        
                        # 10 min buckets (0-9, 10-19... 80-89, 90+)
                        idx_10 = min((minute - 1) // 10, 9) if minute > 0 else 0
                        idx_15 = min((minute - 1) // 15, 5) if minute > 0 else 0
                        
                        # Only count goals SCORED by the team in question
                        # If team was 'home' in history row, it scored 'home' events
                        # If team was 'away' in history row, it scored 'away' events
                        was_home = row.get('is_home', 1)
                        is_scored = (was_home and ev['team'] == 'home') or (not was_home and ev['team'] == 'away')
                        
                        if is_scored:
                            buckets_10[idx_10] += 1
                            buckets_15[idx_15] += 1
                except: continue
                
            return {
                "10m": [round(buckets_10[i] / total_matches, 3) for i in range(10)],
                "15m": [round(buckets_15[i] / total_matches, 3) for i in range(6)]
            }

        return {
            "home": calculate_buckets(home_history),
            "away": calculate_buckets(away_history)
        }
