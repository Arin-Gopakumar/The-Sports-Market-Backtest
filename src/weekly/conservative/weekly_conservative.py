# sports-market-backtest/src/weekly/conservative/weekly_conservative.py
import numpy as np
from ...utils.price_dampening import PriceDampening
from ...utils.z_score_calculator import ZScoreCalculator
from ...utils.stats_calculator import StatsCalculator

class WeeklyConservative:
    """Conservative weekly trading strategy"""
    
    def __init__(self):
        self.dampening = PriceDampening()
        self.z_calculator = ZScoreCalculator()
        self.stats_calc = StatsCalculator()
    
    def calculate_price_change(self, games_data, season_avg, old_price):
        """Calculate price change for conservative weekly strategy"""
        
        # Get last 3 games for weekly projection
        last_3_games = games_data.tail(3)
        
        # Calculate projections (0.6 * season + 0.4 * last 3)
        projected_stats = self.stats_calc.calculate_projected_stats_weekly(
            season_avg, last_3_games
        )
        
        # Calculate actual performance for the week
        actual_stats = self._calculate_weekly_actual(games_data)
        
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
            'z_scores': z_scores,
            'projected_stats': projected_stats,
            'actual_stats': actual_stats
        }
    
    def _calculate_weekly_actual(self, games_data):
        """Calculate actual weekly performance (average of games)"""
        weekly_games = games_data.tail(4)  # Approximate weekly games
        
        return {
            'PTS': weekly_games['PTS'].mean(),
            'REB': weekly_games['REB'].mean(),
            'AST': weekly_games['AST'].mean(),
            'FG%': weekly_games['FG%'].mean(),
            'TO': weekly_games['TO'].mean(),
            'STOCKS': (weekly_games['STL'] + weekly_games['BLK']).mean()
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