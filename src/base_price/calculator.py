# sports-market-backtest/src/base_price/calculator.py
import numpy as np
from scipy.stats import norm

class BasePriceCalculator:
    """Calculate base prices for players"""
    
    def __init__(self):
        self.league_means = {
            'PTS': 13.0,
            'REB': 4.3,
            'AST': 2.8,
            'TO': 2.0,
            'STOCKS': 2.2,
            'FG%': 0.45
        }
        
        self.std_devs = {
            'PTS': 4.0,
            'REB': 2.0,
            'AST': 1.5,
            'TO': 1.0,
            'STOCKS': 1.0,
            'FG%': 0.07
        }
    
    def calculate_prs_non_rookie(self, player_stats):
        """Calculate Player Rating Score for non-rookies"""
        z_scores = {}
        percentiles = {}
        
        # Calculate z-scores
        for stat, value in player_stats.items():
            if stat in self.league_means:
                z = (value - self.league_means[stat]) / self.std_devs[stat]
                z_scores[stat] = z
                percentiles[stat] = norm.cdf(z)
        
        # Calculate PRS with weights
        prs = (0.35 * percentiles.get('PTS', 0.5) +
               0.175 * percentiles.get('AST', 0.5) +
               0.175 * percentiles.get('REB', 0.5) +
               0.125 * percentiles.get('STOCKS', 0.5) +
               0.125 * percentiles.get('FG%', 0.5) +
               0.05 * percentiles.get('TO', 0.5))
        
        base_price = 10 + (prs * 50)
        
        return {
            'base_price': base_price,
            'prs': prs,
            'z_scores': z_scores,
            'percentiles': percentiles
        }
    
    def calculate_drs_rookie(self, draft_pick):
        """Calculate Draft Rating Score for rookies"""
        # DRS = max(0.1, 1 - log2(pick)/log2(60))
        drs = max(0.1, 1 - np.log2(draft_pick) / np.log2(60))
        base_price = 10 + (drs * 25)
        
        return {
            'base_price': base_price,
            'drs': drs,
            'draft_pick': draft_pick
        }