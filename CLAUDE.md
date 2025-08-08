# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sports betting backtest system that analyzes the profitability of NBA player prop bet strategies from the perspective of a sportsbook/bank. The system combines betting odds data (from OddsAPI) with actual player performance data (from Ball Don't Lie API) to evaluate different trading strategies across various timeframes and positions.

## Key Architecture

**Core Components:**
- `main.py`: Entry point that orchestrates the full backtest process across all strategy combinations
- `src/backtesting/backtest_engine.py`: Central engine that matches games, simulates trades, and calculates bank P&L
- `src/utils/data_loader.py`: Handles loading and preprocessing of odds and performance CSV data
- `src/analysis/performance_analyzer.py`: Generates visualizations and summary reports

**Strategy Framework:**
The system uses a modular strategy architecture with three dimensions:
- **Strategies**: `conservative` vs `aggressive` (different risk/dampening profiles)
- **Timeframes**: `intragame`, `weekly`, `monthly` (different volatility assumptions)
- **Positions**: `long` vs `short` (bank's perspective on user trades)

**Data Pipeline:**
1. Raw CSV files in `data/` (separate odds and performance files per player)
2. Consolidated files: `data/consolidated_odds.csv` and `data/consolidated_performance.csv`
3. Game matching logic in backtest engine based on player, date, and opponent
4. Results output to `results/` directory

## Running the Backtest

**Primary Command:**
```bash
python main.py
```

This will:
- Load consolidated data files
- Run all strategy/timeframe/position combinations (2×3×2 = 12 total backtests)
- Generate result CSVs and performance visualizations
- Output summary statistics to console and text file

**Key Output Files:**
- `results/backtest_results.csv` - All trades combined
- `results/backtest_results_long.csv` - Long positions only
- `results/backtest_results_short.csv` - Short positions only
- `results/performance_summary.txt` - Statistical summary
- `results/*.png` - Performance charts (P&L distribution, win rates, player analysis, position comparison)

## Data Processing

**Data Consolidation:**
- Use `consolidate_data.py` to merge individual player CSV files into consolidated datasets
- Handles column name normalization (removes dashes from OddsAPI column headers)
- `check_columns.py` can be used to verify data structure consistency

**Expected Data Structure:**
- Odds data: Game Date, Player Name, Team, Opponent, Market, O/U, American Over/Under Odds
- Performance data: Date, Player, Opponent, PTS, REB, AST, STL, BLK, TO, FG%

## Core Calculation Flow

1. **Game Matching**: Match betting lines to actual performances by player, date, opponent
2. **Z-Score Calculation**: Compare actual vs projected stats using sport-specific standard deviations
3. **PPS Calculation**: Weighted performance score across 6 key stats (PTS, REB, AST, STOCKS, FG%, TO)
4. **Price Dampening**: Apply strategy-specific formulas to convert PPS to price movements
5. **Bank P&L**: Calculate sportsbook profit/loss based on price changes and user position

## Dependencies

The project uses standard Python data science libraries:
- `pandas`, `numpy` for data manipulation
- `matplotlib`, `seaborn` for visualizations
- `scipy` for statistical calculations
- No requirements.txt file present - install dependencies as needed

## Development Notes

- Bank P&L perspective: Positive values mean the bank profits, negative means the bank loses
- The system assumes $25 base price for all trades
- Results are separated by position type since long/short performance can vary significantly
- Strategy modules in subdirectories follow a modular pattern but are primarily coordinated through the backtest engine
- All numeric calculations handle missing/invalid data with appropriate fallbacks