# main.py
import os
import sys
import numpy as np
import pandas as pd

from src.backtesting.backtest_engine import BacktestEngine
from src.utils.data_loader import DataLoader
from src.analysis.performance_analyzer import PerformanceAnalyzer

STAKE_PER_TRADE = 100.0  # dollars per trade for bank/user P&L

def _print_sep(char="=", width=60, title=None):
    if title:
        print("\n" + char * 60)
        print(title)
        print(char * 60 + "\n")
    else:
        print(char * width)

def _summary_stats(series: pd.Series) -> dict:
    if series is None or len(series) == 0:
        return dict(std=np.nan, p01=np.nan, p05=np.nan, p95=np.nan, p99=np.nan, worst=np.nan, best=np.nan)
    return dict(
        std=float(series.std(ddof=1)),
        p01=float(np.percentile(series, 1)),
        p05=float(np.percentile(series, 5)),
        p95=float(np.percentile(series, 95)),
        p99=float(np.percentile(series, 99)),
        worst=float(series.min()),
        best=float(series.max()),
    )

def _percentile_table(df: pd.DataFrame, label: str):
    """Print 1st/5th/95th/99th percentiles by strategy × timeframe (both % and $)."""
    if df.empty:
        return
    print(f"{label}")
    print("-" * 120)
    print(f"{'Strategy':<14}{'Timeframe':<14}{'P1%':>10}{'P5%':>10}{'P95%':>10}{'P99%':>10}{'P1 $':>12}{'P5 $':>12}{'P95 $':>12}{'P99 $':>12}{'Trades':>10}")
    print("-" * 120)
    grp = df.groupby(['strategy', 'timeframe'])
    rows = []
    for (strategy, timeframe), g in grp:
        p1_pct = np.percentile(g['bank_pnl_pct'], 1) if len(g) else np.nan
        p5_pct = np.percentile(g['bank_pnl_pct'], 5) if len(g) else np.nan
        p95_pct = np.percentile(g['bank_pnl_pct'], 95) if len(g) else np.nan
        p99_pct = np.percentile(g['bank_pnl_pct'], 99) if len(g) else np.nan
        p1_usd = np.percentile(g['bank_pnl_dollars'], 1) if len(g) else np.nan
        p5_usd = np.percentile(g['bank_pnl_dollars'], 5) if len(g) else np.nan
        p95_usd = np.percentile(g['bank_pnl_dollars'], 95) if len(g) else np.nan
        p99_usd = np.percentile(g['bank_pnl_dollars'], 99) if len(g) else np.nan
        rows.append((strategy, timeframe, p1_pct, p5_pct, p95_pct, p99_pct, p1_usd, p5_usd, p95_usd, p99_usd, len(g)))
    # Keep output in a fixed order
    order = [('conservative','intragame'), ('conservative','weekly'), ('conservative','monthly'),
             ('aggressive','intragame'),   ('aggressive','weekly'),   ('aggressive','monthly')]
    order_map = {k:i for i,k in enumerate(order)}
    rows.sort(key=lambda r: order_map.get((r[0], r[1]), 999))
    for r in rows:
        print(f"{r[0]:<14}{r[1]:<14}{r[2]:>10.2f}{r[3]:>10.2f}{r[4]:>10.2f}{r[5]:>10.2f}{r[6]:>12.2f}{r[7]:>12.2f}{r[8]:>12.2f}{r[9]:>12.2f}{r[10]:>10}")
    print()

def _summarize_block(df: pd.DataFrame, title: str):
    _print_sep("=", 60, title)
    if df.empty:
        print("No trades.")
        return

    total_trades = len(df)
    bank_win = (df['bank_pnl_pct'] > 0).mean() * 100.0
    user_win = 100.0 - bank_win
    avg_bank_pct = df['bank_pnl_pct'].mean()
    avg_bank_dollars = df['bank_pnl_dollars'].mean()
    total_bank_dollars = df['bank_pnl_dollars'].sum()

    print(f"Total trades: {total_trades}")
    print(f"Bank win rate: {bank_win:.1f}%   (User win rate: {user_win:.1f}%)")
    print(f"Average bank P&L (pct): {avg_bank_pct:.2f}%")
    print(f"Average bank P&L ($/trade @stake={STAKE_PER_TRADE:.0f}): {avg_bank_dollars:.2f}")
    print(f"Total bank P&L ($): {total_bank_dollars:.2f}\n")

    # Distribution stats for the whole block
    dist = _summary_stats(df['bank_pnl_dollars'])
    print("Distribution (bank $ P&L per trade):")
    print(f"  Std Dev: {dist['std']:.2f}")
    print(f"  1st pct: {dist['p01']:.2f}")
    print(f"  5th pct: {dist['p05']:.2f}")
    print(f"  95th pct: {dist['p95']:.2f}")
    print(f"  99th pct: {dist['p99']:.2f}")
    print(f"  Worst trade: {dist['worst']:.2f}")
    print(f"  Best trade: {dist['best']:.2f}\n")

    # Table by strategy & timeframe (averages)
    print("PERFORMANCE BY STRATEGY AND TIMEFRAME:")
    print("-" * 100)
    print(f"{'Strategy':<14}{'Timeframe':<14}{'Trades':>8}{'Win Rate':>12}{'Avg %':>10}{'Avg $':>10}{'Total $':>12}")
    print("-" * 100)
    grouped = df.groupby(['strategy', 'timeframe'], as_index=False).agg(
        trades=('bank_pnl_pct', 'size'),
        win_rate=('bank_pnl_pct', lambda s: (s > 0).mean() * 100.0),
        avg_pct=('bank_pnl_pct', 'mean'),
        avg_usd=('bank_pnl_dollars', 'mean'),
        total_usd=('bank_pnl_dollars', 'sum')
    )
    # Stable row order
    cat_strategy = pd.CategoricalDtype(['conservative','aggressive'], ordered=True)
    cat_timeframe = pd.CategoricalDtype(['intragame','weekly','monthly'], ordered=True)
    grouped['strategy'] = grouped['strategy'].astype(cat_strategy)
    grouped['timeframe'] = grouped['timeframe'].astype(cat_timeframe)
    grouped = grouped.sort_values(['strategy','timeframe'])
    for _, row in grouped.iterrows():
        print(f"{row['strategy']:<14}{row['timeframe']:<14}{int(row['trades']):>8}"
              f"{row['win_rate']:>11.1f}%{row['avg_pct']:>9.2f}%{row['avg_usd']:>10.2f}{row['total_usd']:>12.2f}")
    print()

    # NEW: 5th / 95th percentiles for each of the 6 combos
    _percentile_table(df, "PERCENTILES (by strategy × timeframe) — 5th/95th for % and $")

def main():
    """Run backtests across strategies/timeframes for long & short positions with external projections support."""
    print("Loading data...")
    loader = DataLoader()

    odds_df = loader.load_odds_data('data/consolidated_odds.csv')  # prefers all_odds.csv automatically
    if odds_df is None or odds_df.empty:
        print("ERROR: Could not load odds data.")
        sys.exit(1)

    perf_df = loader.load_performance_data('data/consolidated_performance.csv')  # prefers all_stats.csv automatically
    if perf_df is None or perf_df.empty:
        print("ERROR: Could not load performance data.")
        sys.exit(1)

    # Optional external projections (if present)
    proj_df = loader.load_projections(None)

    print(f"Loaded {len(odds_df)} odds records and {len(perf_df)} performance records\n")

    engine = BacktestEngine(odds_df, perf_df, projections_data=proj_df, stake_per_trade=STAKE_PER_TRADE)

    strategies = ['conservative', 'aggressive']
    timeframes = ['intragame', 'weekly', 'monthly']

    # LONGS
    _print_sep("=", 60, "RUNNING BACKTESTS FOR LONG POSITIONS")
    long_results = []
    for strategy in strategies:
        for timeframe in timeframes:
            print(f"\nRunning long {strategy} {timeframe}...")
            res = engine.run_backtest(strategy=strategy, timeframe=timeframe, position='long')
            if not res.empty:
                long_results.append(res)
    long_df = pd.concat(long_results, ignore_index=True) if long_results else pd.DataFrame()
    _summarize_block(long_df, "LONG POSITIONS SUMMARY")

    # SHORTS
    _print_sep("=", 60, "RUNNING BACKTESTS FOR SHORT POSITIONS")
    short_results = []
    for strategy in strategies:
        for timeframe in timeframes:
            print(f"\nRunning short {strategy} {timeframe}...")
            res = engine.run_backtest(strategy=strategy, timeframe=timeframe, position='short')
            if not res.empty:
                short_results.append(res)
    short_df = pd.concat(short_results, ignore_index=True) if short_results else pd.DataFrame()
    _summarize_block(short_df, "SHORT POSITIONS SUMMARY")

    # Save outputs
    all_df = pd.concat([long_df, short_df], ignore_index=True) if not long_df.empty or not short_df.empty else pd.DataFrame()

    os.makedirs('results', exist_ok=True)
    if not all_df.empty:  all_df.to_csv('results/backtest_results.csv', index=False)
    if not long_df.empty: long_df.to_csv('results/backtest_results_long.csv', index=False)
    if not short_df.empty: short_df.to_csv('results/backtest_results_short.csv', index=False)

    print("\nBacktest complete! Results saved to:")
    if not all_df.empty:  print("- results/backtest_results.csv (all trades)")
    if not long_df.empty: print("- results/backtest_results_long.csv (long positions only)")
    if not short_df.empty: print("- results/backtest_results_short.csv (short positions only)")

    # RECOMMENDATIONS
    _print_sep("=", 80, "RECOMMENDATIONS")
    def best_combo(df: pd.DataFrame):
        if df.empty:
            return None, None
        grp = df.groupby(['strategy','timeframe'])['bank_pnl_dollars'].mean()
        idx = grp.idxmax()
        val = grp.max()
        return idx, val

    long_best, long_best_val   = best_combo(long_df)
    short_best, short_best_val = best_combo(short_df)

    if long_best:
        print(f"Best LONG (avg $/trade): {long_best[0]} {long_best[1]} (${long_best_val:.2f})")
    else:
        print("Best LONG: N/A")

    if short_best:
        print(f"Best SHORT (avg $/trade): {short_best[0]} {short_best[1]} (${short_best_val:.2f})")
    else:
        print("Best SHORT: N/A")

    if long_best_val is not None and short_best_val is not None:
        if long_best_val > short_best_val:
            print(f"\nOverall recommendation: Focus LONG with {long_best[0]} {long_best[1]}")
        elif short_best_val > long_best_val:
            print(f"\nOverall recommendation: Focus SHORT with {short_best[0]} {short_best[1]}")
        else:
            print("\nOverall recommendation: Long and short are tied on avg $/trade.")
    else:
        print("\nOverall recommendation: Insufficient data to compare long vs short.")

    # Optional human-readable text summary
    if not all_df.empty and hasattr(PerformanceAnalyzer, "generate_summary"):
        try:
            analyzer = PerformanceAnalyzer(all_df)
            summary_txt = analyzer.generate_summary()
            with open('results/summary.txt', 'w', encoding='utf-8') as f:
                f.write(summary_txt)
            print("- results/summary.txt (text summary)")
        except Exception as e:
            print(f"(Skipped writing summary.txt due to error: {e})")

if __name__ == '__main__':
    main()
