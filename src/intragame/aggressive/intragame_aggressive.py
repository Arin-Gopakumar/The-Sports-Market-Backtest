# sports-market-backtest/src/intragame/aggressive/intragame_aggressive.py
import numpy as np
from ...utils.price_dampening import PriceDampening
from ...utils.z_score_calculator import ZScoreCalculator

class IntragameAggressive:
    """Aggressive intragame trading strategy"""
    
    def __init__(self):
        self.dampening = PriceDampening()
        self.z_calculator = ZScoreCalculator()
    
    def calculate_price_change(self, actual_stats, projected_stats, old_price):
        """Calculate price change for aggressive intragame strategy"""
        
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
        
        # Apply aggressive dampening
        dampened_delta = self._apply_aggressive_dampening(pps)
        
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
    
    def _apply_aggressive_dampening(self, raw_delta):
        """Apply aggressive dampening formula for intragame"""
        if raw_delta >= 0:  # Upside
            dampened = min(
                raw_delta / np.sqrt(max(1 - 0.216 * raw_delta**2, 0.001)),
                20
            )
        else:  # Downside
            dampened = -1 * abs(raw_delta**2) / (abs(raw_delta**2) + 0.18)
        
        return dampened