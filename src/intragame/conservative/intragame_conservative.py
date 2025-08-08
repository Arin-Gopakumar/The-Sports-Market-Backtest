# sports-market-backtest/src/intragame/conservative/intragame_conservative.py
import numpy as np
from ...utils.price_dampening import PriceDampening
from ...utils.z_score_calculator import ZScoreCalculator

class IntragameConservative:
    """Conservative intragame trading strategy"""
    
    def __init__(self):
        self.dampening = PriceDampening()
        self.z_calculator = ZScoreCalculator()
    
    def calculate_price_change(self, actual_stats, projected_stats, old_price):
        """Calculate price change for conservative intragame strategy"""
        
        # Calculate z-scores
        z_scores = {}
        for stat in ['PTS', 'REB', 'AST', 'FG%', 'TO', 'STOCKS']:
            if stat in actual_stats and stat in projected_stats:
                std_dev = self.z_calculator.calculate_standard_deviation(
                    projected_stats[stat], stat
                )
                z_scores[stat] = self.z_calculator.calculate_z_score(
                    actual_stats[stat], 
                    projected_stats[stat], 
                    std_dev, 
                    stat
                )
        
        # Calculate PPS
        pps = self.dampening.calculate_pps(z_scores)
        
        # Apply conservative dampening
        dampened_delta = self._apply_conservative_dampening(pps)
        
        # Calculate new price
        new_price = self.dampening.calculate_final_price(old_price, dampened_delta)
        
        return {
            'old_price': old_price,
            'new_price': new_price,
            'price_change_pct': ((new_price - old_price) / old_price) * 100,
            'pps': pps,
            'dampened_delta': dampened_delta,
            'z_scores': z_scores
        }
    
    def _apply_conservative_dampening(self, raw_delta):
        """Apply conservative dampening formula"""
        if raw_delta >= 0:  # Upside
            if 0 <= raw_delta <= 2:
                dampened = raw_delta / np.sqrt(1 + 4 * raw_delta**2)
            else:  # raw_delta > 2
                dampened = 1.5 / (1 + np.exp(-6 * (raw_delta - 2)))
        else:  # Downside
            dampened = raw_delta / np.sqrt(1 + 2.5 * raw_delta**2)
        
        return dampened