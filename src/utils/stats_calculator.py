# sports-market-backtest/src/utils/stats_calculator.py
import numpy as np
import pandas as pd

class StatsCalculator:
    """Calculate various statistical measures for player performance"""
    
    @staticmethod
    def calculate_base_stats(games_df, num_games=25):
        """Calculate mean and standard deviation from recent games"""
        if len(games_df) < num_games:
            num_games = len(games_df)
        
        recent_games = games_df.tail(num_games)
        
        stats = {}
        for col in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TO', 'FG%']:
            if col in recent_games.columns:
                stats[f'{col}_mean'] = recent_games[col].mean()
                stats[f'{col}_std'] = recent_games[col].std()
        
        # Calculate STOCKS (Steals + Blocks)
        if 'STL' in recent_games.columns and 'BLK' in recent_games.columns:
            stocks = recent_games['STL'] + recent_games['BLK']
            stats['STOCKS_mean'] = stocks.mean()
            stats['STOCKS_std'] = stocks.std()
        
        return stats
    
    @staticmethod
    def calculate_projected_stats_intragame(last_5_games):
        """For intragame: FG%, TO, STL+BLK use average of last 5 games"""
        projections = {}
        
        if len(last_5_games) >= 5:
            recent = last_5_games.tail(5)
        else:
            recent = last_5_games
        
        projections['FG%'] = recent['FG%'].mean()
        projections['TO'] = recent['TO'].mean()
        projections['STOCKS'] = (recent['STL'] + recent['BLK']).mean()
        
        return projections
    
    @staticmethod
    def calculate_projected_stats_weekly(season_avg, last_3_games):
        """Weekly: 0.6 × Season Avg + 0.4 × Last 3 Games Avg"""
        projections = {}
        
        if len(last_3_games) >= 3:
            recent = last_3_games.tail(3)
        else:
            recent = last_3_games
        
        for stat in ['PTS', 'REB', 'AST', 'FG%', 'TO']:
            season_val = season_avg.get(f'{stat}_mean', 0)
            recent_val = recent[stat].mean() if stat in recent.columns else 0
            projections[stat] = 0.6 * season_val + 0.4 * recent_val
        
        # Handle STOCKS separately
        season_stocks = season_avg.get('STOCKS_mean', 0)
        recent_stocks = (recent['STL'] + recent['BLK']).mean()
        projections['STOCKS'] = 0.6 * season_stocks + 0.4 * recent_stocks
        
        return projections
    
    @staticmethod
    def calculate_projected_stats_monthly(season_avg, last_12_games):
        """Monthly: 0.6 × Season Avg + 0.4 × Last 12 Games Avg"""
        projections = {}
        
        if len(last_12_games) >= 12:
            recent = last_12_games.tail(12)
        else:
            recent = last_12_games
        
        for stat in ['PTS', 'REB', 'AST', 'FG%', 'TO']:
            season_val = season_avg.get(f'{stat}_mean', 0)
            recent_val = recent[stat].mean() if stat in recent.columns else 0
            projections[stat] = 0.6 * season_val + 0.4 * recent_val
        
        # Handle STOCKS separately
        season_stocks = season_avg.get('STOCKS_mean', 0)
        recent_stocks = (recent['STL'] + recent['BLK']).mean()
        projections['STOCKS'] = 0.6 * season_stocks + 0.4 * recent_stocks
        
        return projections