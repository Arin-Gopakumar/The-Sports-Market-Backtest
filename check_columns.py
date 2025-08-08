# check_columns.py
import pandas as pd

print("Checking column names in consolidated files...\n")

# Check odds data
print("ODDS DATA COLUMNS:")
print("-" * 40)
try:
    odds_df = pd.read_csv('data/consolidated_odds.csv')
    print(f"Columns: {list(odds_df.columns)}")
    print(f"Shape: {odds_df.shape}")
    print("\nFirst few rows:")
    print(odds_df.head(2))
except Exception as e:
    print(f"Error reading odds data: {e}")

print("\n\nPERFORMANCE DATA COLUMNS:")
print("-" * 40)
try:
    perf_df = pd.read_csv('data/consolidated_performance.csv')
    print(f"Columns: {list(perf_df.columns)}")
    print(f"Shape: {perf_df.shape}")
    print("\nFirst few rows:")
    print(perf_df.head(2))
except Exception as e:
    print(f"Error reading performance data: {e}")