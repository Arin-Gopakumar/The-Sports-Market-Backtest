# sports-market-backtest/src/utils/odds_converter.py
import numpy as np
from scipy.stats import norm

class OddsConverter:
    """Convert American odds to probabilities and derive statistical parameters"""
    
    @staticmethod
    def american_to_probability(odds):
        """Convert American odds to implied probability"""
        if odds < 0:
            return -odds / (-odds + 100)
        else:
            return 100 / (odds + 100)
    
    @staticmethod
    def remove_vig(prob_over, prob_under):
        """Remove vig to get fair probabilities"""
        total = prob_over + prob_under
        return prob_over / total, prob_under / total
    
    @staticmethod
    def probability_to_zscore(probability):
        """Convert probability to z-score using inverse normal CDF"""
        return norm.ppf(probability)
    
    @staticmethod
    def derive_parameters(line, prob_fair):
        """
        Derive mean and standard deviation from betting line and fair probability
        
        For an over/under line L with fair probability P(Over):
        - Mean μ = L - z*σ where z = Φ^(-1)(P(Over))
        - We need to estimate σ based on the stat type
        """
        z_score = OddsConverter.probability_to_zscore(prob_fair)
        
        # Estimate sigma based on typical variance for each stat
        # These are approximations that would need calibration
        estimated_sigmas = {
            'points': 5.0,    # Points typically vary by ~5
            'rebounds': 2.5,  # Rebounds vary by ~2.5
            'assists': 2.0,   # Assists vary by ~2
            'stocks': 1.0,    # Steals+blocks vary by ~1
            'turnovers': 1.0, # Turnovers vary by ~1
            'fg_pct': 0.07    # FG% varies by ~7%
        }
        
        # For now, assume points (will be determined by market type)
        sigma = estimated_sigmas.get('points', 3.0)
        
        # Solve for mu: L = mu + z*sigma => mu = L - z*sigma
        mu = line - z_score * sigma
        
        return mu, sigma