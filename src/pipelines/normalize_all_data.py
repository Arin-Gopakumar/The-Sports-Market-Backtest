# src/pipelines/normalize_all_data.py
"""
Normalize all_odds.csv and all_stats.csv (each mixing real + synthetic rows)
into the canonical files the backtester expects:
  - data/consolidated_odds.csv
  - data/consolidated_performance.csv

Usage (from project root):
    python -m src.pipelines.normalize_all_data
"""
import os
import re
import sys
import pandas as pd
from pathlib import Path

RAW_ODDS = Path("data/all_odds.csv")
RAW_STATS = Path("data/all_stats.csv")

OUT_ODDS = Path("data/consolidated_odds.csv")
OUT_STATS = Path("data/consolidated_performance.csv")

# ---------- helpers ----------

def _first_col(df, *candidates):
    for cand in candidates:
        for c in df.columns:
            if c.strip().lower() == cand.strip().lower() or cand.strip().lower() in c.strip().lower():
                return c
    return None

def _parse_date_series(s):
    return pd.to_datetime(s, errors="coerce")

def _as_float(s):
    return pd.to_numeric(s, errors="coerce")

def _find_source_column(df):
    for c in df.columns:
        if df[c].astype(str).str.contains(r"\bSYNTHETIC\b", case=False, na=False).any():
            return c
    return None

def _derive_source(df):
    c = _find_source_column(df)
    if c:
        src = df[c].astype(str).str.upper().str.contains("SYNTHETIC")
        return pd.Series(["SYNTHETIC" if v else "REAL" for v in src], index=df.index)
    # fallback: mark everything REAL if no hint
    return pd.Series(["REAL"] * len(df), index=df.index)

def _split_over_under_prob(text):
    """
    Parse strings like 'Over: 47.6% / Under: 58.3%' -> (0.476, 0.583)
    """
    if not isinstance(text, str):
        return (None, None)
    m = re.findall(r"(Over|Under)\s*:\s*([0-9.]+)%", text, flags=re.I)
    if not m:
        return (None, None)
    d = {k.lower(): float(v) / 100.0 for (k, v) in m}
    return d.get("over"), d.get("under")

# ---------- normalizers ----------

def normalize_odds(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    c_date     = _first_col(df, "Game Date", "Date")
    c_player   = _first_col(df, "Player Name", "Player")
    c_team     = _first_col(df, "Team")
    c_opp      = _first_col(df, "Opponent")
    c_market   = _first_col(df, "Market", "Stat", "Category")
    c_line     = _first_col(df, "O/U", "Line", "Number")
    c_over_od  = _first_col(df, "American Over Odds", "Over Odds", "Over")
    c_under_od = _first_col(df, "American Under Odds", "Under Odds", "Under")
    c_probtxt  = _first_col(df, "Over/Under", "Implied Probability", "Probability", "Over:")

    out = pd.DataFrame()
    out["Game Date"] = _parse_date_series(df[c_date]) if c_date else pd.NaT
    out["Player Name"] = df[c_player] if c_player else None
    out["Team"] = df[c_team] if c_team else None
    out["Opponent"] = df[c_opp] if c_opp else None
    out["Market"] = df[c_market] if c_market else None
    out["O/U"] = _as_float(df[c_line]) if c_line else None
    out["American Over Odds"] = _as_float(df[c_over_od]) if c_over_od else None
    out["American Under Odds"] = _as_float(df[c_under_od]) if c_under_od else None

    if c_probtxt:
        over_p, under_p = [], []
        for v in df[c_probtxt].tolist():
            p_over, p_under = _split_over_under_prob(v)
            over_p.append(p_over)
            under_p.append(p_under)
        out["Over Prob"] = over_p
        out["Under Prob"] = under_p

    out["Source"] = _derive_source(df)

    for c in ["Player Name", "Team", "Opponent", "Market", "Source"]:
        out[c] = out[c].astype(str).str.strip()

    out = out.sort_values(["Player Name", "Game Date", "Market"], kind="mergesort").reset_index(drop=True)
    return out

def normalize_stats(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    c_date   = _first_col(df, "Date", "Game Date")
    c_player = _first_col(df, "Player", "Player Name")
    c_opp    = _first_col(df, "Opponent")
    c_pts = _first_col(df, "PTS", "Points")
    c_reb = _first_col(df, "REB", "Rebounds")
    c_ast = _first_col(df, "AST", "Assists")
    c_stl = _first_col(df, "STL", "Steals")
    c_blk = _first_col(df, "BLK", "Blocks")
    c_to  = _first_col(df, "TO", "Turnovers")
    c_fg  = _first_col(df, "FG%", "FGP", "Field Goal Percentage")

    out = pd.DataFrame()
    out["Date"]   = _parse_date_series(df[c_date]) if c_date else pd.NaT
    out["Player"] = df[c_player] if c_player else None
    out["Opponent"] = df[c_opp] if c_opp else None
    out["PTS"] = _as_float(df[c_pts]) if c_pts else None
    out["REB"] = _as_float(df[c_reb]) if c_reb else None
    out["AST"] = _as_float(df[c_ast]) if c_ast else None
    out["STL"] = _as_float(df[c_stl]) if c_stl else 0.0
    out["BLK"] = _as_float(df[c_blk]) if c_blk else 0.0
    out["TO"]  = _as_float(df[c_to])  if c_to  else None
    out["FG%"] = _as_float(df[c_fg])  if c_fg  else None

    out["Source"] = _derive_source(df)

    for c in ["Player", "Opponent", "Source"]:
        out[c] = out[c].astype(str).str.strip()

    out = out.sort_values(["Player", "Date"], kind="mergesort").reset_index(drop=True)
    return out

def main():
    if not RAW_ODDS.exists():
        print(f"ERROR: {RAW_ODDS} not found. Place all_odds.csv under data/", file=sys.stderr)
        sys.exit(1)
    if not RAW_STATS.exists():
        print(f"ERROR: {RAW_STATS} not found. Place all_stats.csv under data/", file=sys.stderr)
        sys.exit(1)

    print("Normalizing odds...")
    odds = normalize_odds(RAW_ODDS)
    print(f" -> {len(odds)} rows")
    OUT_ODDS.parent.mkdir(parents=True, exist_ok=True)
    odds.to_csv(OUT_ODDS, index=False)
    print(f"Wrote {OUT_ODDS}")

    print("Normalizing stats...")
    stats = normalize_stats(RAW_STATS)
    print(f" -> {len(stats)} rows")
    OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(OUT_STATS, index=False)
    print(f"Wrote {OUT_STATS}")

if __name__ == "__main__":
    main()
