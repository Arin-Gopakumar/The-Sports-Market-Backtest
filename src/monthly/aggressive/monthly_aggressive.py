# sports-market-backtest/src/monthly/aggressive/monthly_aggressive.py
import numpy as np
from ...utils.price_dampening import PriceDampening
from ...utils.z_score_calculator import ZScoreCalculator
from ...utils.stats_calculator import StatsCalculator

class MonthlyAggressive:
    """Aggressive monthly trading strategy"""
    
    def __init__(self):
        self.dampening = PriceDampening()
        self.z_calculator = ZScoreCalculator()
        self.stats_calc = StatsCalculator()
    
    def calculate_price_change(self, games_data, season_avg, old_price):
        """Calculate price change for aggressive monthly strategy"""
        
        # Get last 12 games for monthly projection
        last_12_games = games_data.tail(12)
        
        # Calculate projections (0.6 * season + 0.4 * last 12)
        projected_stats = self.stats_calc.calculate_projected_stats_monthly(
            season_avg, last_12_games
        )
        
        # Calculate actual performance for the month
        actual_stats = self._calculate_monthly_actual(games_data)
        
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
            'z_scores': z_scores,
            'projected_stats': projected_stats,
            'actual_stats': actual_stats
        }
    
    def _calculate_monthly_actual(self, games_data):
        """Calculate actual monthly performance (average of games)"""
        monthly_games = games_data.tail(12)  # Approximate monthly games
        
        return {
            'PTS': monthly_games['PTS'].mean(),
            'REB': monthly_games['REB'].mean(),
            'AST': monthly_games['AST'].mean(),
            'FG%': monthly_games['FG%'].mean(),
            'TO': monthly_games['TO'].mean(),
            'STOCKS': (monthly_games['STL'] + monthly_games['BLK']).mean()
        }
    
    def _apply_aggressive_dampening(self, raw_delta):
        """Apply aggressive dampening formula for monthly"""
        if raw_delta >= 0:  # Upside
            dampened = min(
                raw_delta / np.sqrt(max(1 - 0.216 * raw_delta**2, 0.001)), 
                30
                )
        else:  # Downside
            dampened = -1 * abs(raw_delta**2) / (abs(raw_delta**2)+ 0.18)
        
        return dampened