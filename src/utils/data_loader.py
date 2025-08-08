# src/utils/data_loader.py
import os
from pathlib import Path
import pandas as pd
from typing import Union

class DataLoader:
    """Load and preprocess data from CSV files (real + synthetic + optional projections)."""

    # ---------- public API ----------

    def load_odds_data(self, filepath: Union[str, None] = None, prefer_all: bool = True) -> Union[pd.DataFrame, None]:
        # Prefer all_odds.csv
        if prefer_all:
            df = self._try_load_all_odds("data/all_odds.csv")
            if df is not None:
                print("Loading odds data from: data/all_odds.csv (normalized)")
                return df

        # Fall back to consolidated files
        for p in self._candidates(filepath, "data/consolidated_odds.csv", "consolidated_odds.csv", "data/oddsapi_data.csv"):
            if Path(p).exists():
                print(f"Loading odds data from: {p}")
                return pd.read_csv(p)

        print("Odds data not found.")
        return None

    def load_performance_data(self, filepath: Union[str, None] = None, prefer_all: bool = True) -> Union[pd.DataFrame, None]:
        if prefer_all:
            df = self._try_load_all_stats("data/all_stats.csv")
            if df is not None:
                print("Loading performance data from: data/all_stats.csv (normalized)")
                return df

        for p in self._candidates(filepath, "data/consolidated_performance.csv", "consolidated_performance.csv"):
            if Path(p).exists():
                print(f"Loading performance data from: {p}")
                return pd.read_csv(p)

        print("Performance data not found.")
        return None

    def load_projections(self, filepath: Union[str, None] = None) -> Union[pd.DataFrame, None]:
        """Optional external projections file.
        Expected columns (flexible): Date, Player, Proj_* stats, or per-timeframe ProjWeekly_*, ProjMonthly_*.
        """
        for p in self._candidates(filepath, "data/projected_stats.csv", "projected_stats.csv", "data/all_projections.csv"):
            if p and Path(p).exists():
                print(f"Loading projections from: {p}")
                df = pd.read_csv(p)
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                return df
        print("Projections file not found (optional).")
        return None

    # ---------- helpers ----------

    @staticmethod
    def _candidates(*paths):
        return [p for p in paths if p]

    @staticmethod
    def _first_col(df, *candidates):
        for cand in candidates:
            for c in df.columns:
                if c.strip().lower() == cand.strip().lower() or cand.strip().lower() in c.strip().lower():
                    return c
        return None

    @staticmethod
    def _parse_date_series(s):
        return pd.to_datetime(s, errors="coerce")

    @staticmethod
    def _as_float(s):
        return pd.to_numeric(s, errors="coerce")

    @staticmethod
    def _find_source_column(df):
        for c in df.columns:
            if df[c].astype(str).str.contains(r"\bSYNTHETIC\b", case=False, na=False).any():
                return c
        return None

    def _derive_source(self, df):
        c = self._find_source_column(df)
        if c:
            src = df[c].astype(str).str.upper().str.contains("SYNTHETIC")
            return pd.Series(["SYNTHETIC" if v else "REAL" for v in src], index=df.index)
        return pd.Series(["REAL"] * len(df), index=df.index)

    # ---------- inline normalizers so we can run without the pipeline ----------

    def _try_load_all_odds(self, path: str) -> Union[pd.DataFrame, None]:
        p = Path(path)
        if not p.exists():
            return None
        raw = pd.read_csv(p)

        c_date     = self._first_col(raw, "Game Date", "Date")
        c_player   = self._first_col(raw, "Player Name", "Player")
        c_team     = self._first_col(raw, "Team")
        c_opp      = self._first_col(raw, "Opponent")
        c_market   = self._first_col(raw, "Market", "Stat", "Category")
        c_line     = self._first_col(raw, "O/U", "Line", "Number")
        c_over_od  = self._first_col(raw, "American Over Odds", "Over Odds", "Over")
        c_under_od = self._first_col(raw, "American Under Odds", "Under Odds", "Under")

        out = pd.DataFrame()
        out["Game Date"] = self._parse_date_series(raw[c_date]) if c_date else pd.NaT
        out["Player Name"] = raw[c_player] if c_player else None
        out["Team"] = raw[c_team] if c_team else None
        out["Opponent"] = raw[c_opp] if c_opp else None
        out["Market"] = raw[c_market] if c_market else None
        out["O/U"] = self._as_float(raw[c_line]) if c_line else None
        out["American Over Odds"] = self._as_float(raw[c_over_od]) if c_over_od else None
        out["American Under Odds"] = self._as_float(raw[c_under_od]) if c_under_od else None
        out["Source"] = self._derive_source(raw)

        for c in ["Player Name", "Team", "Opponent", "Market", "Source"]:
            out[c] = out[c].astype(str).str.strip()

        out = out.sort_values(["Player Name", "Game Date", "Market"], kind="mergesort").reset_index(drop=True)
        return out

    def _try_load_all_stats(self, path: str) -> Union[pd.DataFrame, None]:
        p = Path(path)
        if not p.exists():
            return None
        raw = pd.read_csv(p)

        c_date   = self._first_col(raw, "Date", "Game Date")
        c_player = self._first_col(raw, "Player", "Player Name")
        c_opp    = self._first_col(raw, "Opponent")
        c_pts = self._first_col(raw, "PTS", "Points")
        c_reb = self._first_col(raw, "REB", "Rebounds")
        c_ast = self._first_col(raw, "AST", "Assists")
        c_stl = self._first_col(raw, "STL", "Steals")
        c_blk = self._first_col(raw, "BLK", "Blocks")
        c_to  = self._first_col(raw, "TO", "Turnovers")
        c_fg  = self._first_col(raw, "FG%", "FGP", "Field Goal Percentage")

        out = pd.DataFrame()
        out["Date"]   = self._parse_date_series(raw[c_date]) if c_date else pd.NaT
        out["Player"] = raw[c_player] if c_player else None
        out["Opponent"] = raw[c_opp] if c_opp else None
        out["PTS"] = self._as_float(raw[c_pts]) if c_pts else None
        out["REB"] = self._as_float(raw[c_reb]) if c_reb else None
        out["AST"] = self._as_float(raw[c_ast]) if c_ast else None
        out["STL"] = self._as_float(raw[c_stl]) if c_stl else 0.0
        out["BLK"] = self._as_float(raw[c_blk]) if c_blk else 0.0
        out["TO"]  = self._as_float(raw[c_to])  if c_to  else None
        out["FG%"] = self._as_float(raw[c_fg])  if c_fg  else None
        out["Source"] = self._derive_source(raw)

        for c in ["Player", "Opponent", "Source"]:
            out[c] = out[c].astype(str).str.strip()

        out = out.sort_values(["Player", "Date"], kind="mergesort").reset_index(drop=True)
        return out
