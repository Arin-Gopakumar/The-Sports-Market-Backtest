# src/analysis/performance_analyzer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceAnalyzer:
    """Analyze backtest performance and generate reports"""
    
    def __init__(self, results_df):
        self.results = results_df
    
    def generate_full_report(self, output_dir='results/'):
        """Generate comprehensive performance report"""
        
        # Create visualizations
        self._plot_pnl_by_strategy()
        self._plot_win_rates()
        self._plot_player_performance()
        self._plot_position_analysis()
        
        # Generate statistical summary
        summary = self._generate_summary_stats()
        
        # Save summary to file
        with open(f'{output_dir}/performance_summary.txt', 'w') as f:
            f.write("SPORTS MARKET BACKTEST PERFORMANCE SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(summary)
        
        return summary
    
    def _plot_pnl_by_strategy(self):
        """Plot P&L distribution by strategy and timeframe"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Bank P&L Distribution by Strategy and Timeframe', fontsize=16)
        
        strategies = ['conservative', 'aggressive']
        timeframes = ['intragame', 'weekly', 'monthly']
        
        for i, strategy in enumerate(strategies):
            for j, timeframe in enumerate(timeframes):
                ax = axes[i, j]
                data = self.results[
                    (self.results['strategy'] == strategy) & 
                    (self.results['timeframe'] == timeframe)
                ]['bank_pnl_pct']
                
                if len(data) > 0:
                    ax.hist(data, bins=30, alpha=0.7, color='blue' if strategy == 'conservative' else 'red')
                    ax.axvline(data.mean(), color='black', linestyle='--', label=f'Mean: {data.mean():.2f}%')
                    ax.set_title(f'{strategy.capitalize()} - {timeframe.capitalize()}')
                    ax.set_xlabel('Bank P&L %')
                    ax.set_ylabel('Frequency')
                    ax.legend()
        
        plt.tight_layout()
        plt.savefig('results/pnl_distribution.png')
        plt.close()
    
    def _plot_win_rates(self):
        """Plot win rates comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate win rates
        win_rates = []
        labels = []
        
        for strategy in ['conservative', 'aggressive']:
            for timeframe in ['intragame', 'weekly', 'monthly']:
                subset = self.results[
                    (self.results['strategy'] == strategy) & 
                    (self.results['timeframe'] == timeframe)
                ]
                if len(subset) > 0:
                    win_rate = (subset['bank_pnl_pct'] > 0).mean() * 100
                    win_rates.append(win_rate)
                    labels.append(f'{strategy[:4]}-{timeframe[:4]}')
        
        # Create bar plot
        x = np.arange(len(labels))
        colors = ['blue' if 'cons' in label else 'red' for label in labels]
        
        ax.bar(x, win_rates, color=colors, alpha=0.7)
        ax.set_xlabel('Strategy-Timeframe')
        ax.set_ylabel('Bank Win Rate %')
        ax.set_title('Bank Win Rates by Strategy and Timeframe')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.axhline(50, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('results/win_rates.png')
        plt.close()
    
    def _plot_player_performance(self):
        """Plot performance by player"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate average P&L by player
        player_pnl = self.results.groupby('player')['bank_pnl_pct'].agg(['mean', 'count'])
        player_pnl = player_pnl[player_pnl['count'] >= 10]  # Filter for significance
        player_pnl = player_pnl.sort_values('mean', ascending=True)
        
        if len(player_pnl) > 0:
            player_pnl['mean'].plot(kind='barh', ax=ax)
            ax.set_xlabel('Average Bank P&L %')
            ax.set_ylabel('Player')
            ax.set_title('Average Bank P&L by Player')
            ax.axvline(0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('results/player_performance.png')
        plt.close()
    
    def _plot_position_analysis(self):
        """Plot long vs short position analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # P&L by position
        for position in ['long', 'short']:
            data = self.results[self.results['position'] == position]['bank_pnl_pct']
            ax1.hist(data, bins=30, alpha=0.5, label=position)
        
        ax1.set_xlabel('Bank P&L %')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Bank P&L Distribution: Long vs Short')
        ax1.legend()
        
        # Win rate by position
        positions = ['long', 'short']
        win_rates = []
        avg_pnls = []
        
        for position in positions:
            subset = self.results[self.results['position'] == position]
            win_rates.append((subset['bank_pnl_pct'] > 0).mean() * 100)
            avg_pnls.append(subset['bank_pnl_pct'].mean())
        
        x = np.arange(len(positions))
        width = 0.35
        
        ax2.bar(x - width/2, win_rates, width, label='Win Rate %', alpha=0.7)
        ax2.bar(x + width/2, avg_pnls, width, label='Avg P&L %', alpha=0.7)
        
        ax2.set_xlabel('Position Type')
        ax2.set_ylabel('Percentage')
        ax2.set_title('Bank Performance by Position Type')
        ax2.set_xticks(x)
        ax2.set_xticklabels(positions)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('results/position_analysis.png')
        plt.close()
    
    def _generate_summary_stats(self):
        """Generate summary statistics"""
        summary = []
        
        # Overall stats
        summary.append("OVERALL STATISTICS")
        summary.append("-" * 40)
        summary.append(f"Total trades: {len(self.results)}")
        summary.append(f"Overall bank win rate: {(self.results['bank_pnl_pct'] > 0).mean() * 100:.1f}%")
        summary.append(f"Average bank P&L: {self.results['bank_pnl_pct'].mean():.2f}%")
        summary.append(f"Median bank P&L: {self.results['bank_pnl_pct'].median():.2f}%")
        summary.append(f"Std Dev: {self.results['bank_pnl_pct'].std():.2f}%")
        summary.append("")
        
        # By strategy and timeframe
        summary.append("PERFORMANCE BY STRATEGY AND TIMEFRAME")
        summary.append("-" * 40)
        
        for strategy in ['conservative', 'aggressive']:
            for timeframe in ['intragame', 'weekly', 'monthly']:
                subset = self.results[
                    (self.results['strategy'] == strategy) & 
                    (self.results['timeframe'] == timeframe)
                ]
                
                if len(subset) > 0:
                    win_rate = (subset['bank_pnl_pct'] > 0).mean() * 100
                    avg_pnl = subset['bank_pnl_pct'].mean()
                    sharpe = avg_pnl / subset['bank_pnl_pct'].std() if subset['bank_pnl_pct'].std() > 0 else 0
                    
                    summary.append(f"\n{strategy.upper()} - {timeframe.upper()}:")
                    summary.append(f"  Trades: {len(subset)}")
                    summary.append(f"  Win Rate: {win_rate:.1f}%")
                    summary.append(f"  Avg P&L: {avg_pnl:.2f}%")
                    summary.append(f"  Sharpe Ratio: {sharpe:.2f}")
        
        # Best strategy
        summary.append("\n" + "="*40)
        summary.append("RECOMMENDED STRATEGY")
        summary.append("="*40)
        
        best_combo = self.results.groupby(['strategy', 'timeframe'])['bank_pnl_pct'].mean().idxmax()
        best_avg_pnl = self.results.groupby(['strategy', 'timeframe'])['bank_pnl_pct'].mean().max()
        
        summary.append(f"Best performing: {best_combo[0]} - {best_combo[1]}")
        summary.append(f"Average bank P&L: {best_avg_pnl:.2f}%")
        
        return "\n".join(summary)