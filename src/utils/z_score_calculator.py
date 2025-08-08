# sports-market-backtest/src/utils/z_score_calculator.py
import numpy as np

class ZScoreCalculator:
    """Calculate z-scores for player performance"""
    
    # League averages from the formula sheet
    LEAGUE_MEANS = {
        'PTS': 13.0,
        'REB': 4.3,
        'AST': 2.8,
        'TO': 2.0,
        'STOCKS': 2.2,
        'FG%': 0.45
    }
    
    # Standard deviations for base price calculation
    BASE_PRICE_STD = {
        'PTS': 4.0,
        'REB': 2.0,
        'AST': 1.5,
        'TO': 1.0,
        'STOCKS': 1.0,
        'FG%': 0.07
    }
    
    @staticmethod
    def calculate_z_score(actual, projected, std_dev, stat_type='default'):
        """Calculate z-score for a stat"""
        if stat_type == 'TO':  # Turnovers are inverted
            return (projected - actual) / std_dev
        else:
            return (actual - projected) / std_dev
    
    @staticmethod
    def calculate_standard_deviation(projected_stat, stat_type, num_games=25):
        """
        Calculate standard deviation using the formula from the sheet
        Note: The formula uses last 10 games but mentions 25 games in text
        """
        # For rookies with < 25 games, apply multipliers
        multiplier = 1.0
        if num_games < 5:
            return None  # Don't list until 5 games
        elif 5 <= num_games < 10:
            multiplier = 0.85
        elif 10 <= num_games < 25:
            multiplier = 0.95
        
        # Base calculation would use actual game variance
        # For now, using approximations
        base_std = {
            'PTS': np.sqrt(projected_stat) * 1.2,
            'REB': np.sqrt(projected_stat) * 1.0,
            'AST': np.sqrt(projected_stat) * 0.9,
            'TO': np.sqrt(projected_stat) * 0.8,
            'STOCKS': np.sqrt(projected_stat) * 0.8,
            'FG%': 0.07  # Fixed for FG%
        }
        
        return base_std.get(stat_type, 1.0) * multiplier