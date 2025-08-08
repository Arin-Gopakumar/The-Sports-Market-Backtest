# src/backtesting/backtest_engine.py
import math
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

from ..utils.price_dampening import PriceDampening

STATS = ['PTS', 'REB', 'AST', 'FG%', 'TO', 'STOCKS']

class BacktestEngine:
    """Backtesting engine (uses real+synthetic data and optional external projections)."""

    def __init__(self,
                 odds_data: pd.DataFrame,
                 performance_data: pd.DataFrame,
                 projections_data: Optional[pd.DataFrame] = None,
                 stake_per_trade: float = 100.0):
        # Deduplicate oddities
        if 'Player Name' in odds_data.columns and odds_data.columns.tolist().count('Player Name') > 1:
            odds_data = odds_data.loc[:, ~odds_data.columns.duplicated()]

        self.odds_data = odds_data.copy()
        self.performance_data = performance_data.copy()
        self.projections_data = projections_data.copy() if projections_data is not None else None

        # Dates
        if 'Game Date' in self.odds_data.columns:
            self.odds_data['Game Date'] = pd.to_datetime(self.odds_data['Game Date'], errors='coerce')
        if 'Date' in self.performance_data.columns:
            self.performance_data['Date'] = pd.to_datetime(self.performance_data['Date'], errors='coerce')
        if self.projections_data is not None and 'Date' in self.projections_data.columns:
            self.projections_data['Date'] = pd.to_datetime(self.projections_data['Date'], errors='coerce')

        # Standardize perf column
        if 'Player' not in self.performance_data.columns and 'Player Name' in self.performance_data.columns:
            self.performance_data = self.performance_data.rename(columns={'Player Name': 'Player'})

        # STOCKS helper
        if 'STOCKS' not in self.performance_data.columns:
            if {'STL', 'BLK'}.issubset(self.performance_data.columns):
                self.performance_data['STOCKS'] = self.performance_data['STL'].astype(float) + self.performance_data['BLK'].astype(float)
            else:
                self.performance_data['STOCKS'] = np.nan

        # Numeric coercion
        for col in STATS:
            if col in self.performance_data.columns:
                self.performance_data[col] = pd.to_numeric(self.performance_data[col], errors='coerce')

        if self.projections_data is not None:
            self._normalize_projection_columns()

        self.stake_per_trade = float(stake_per_trade)

    # ---------- Matching ----------

    def match_games(self) -> List[Dict]:
        odds = self.odds_data.copy()
        perf = self.performance_data.copy()

        def norm(s): return str(s).strip().lower()
        odds['_player_key'] = odds['Player Name'].map(norm)
        perf['_player_key']  = perf['Player'].map(norm)

        merged = pd.merge(
            odds,
            perf,
            left_on=['_player_key', 'Game Date'],
            right_on=['_player_key', 'Date'],
            how='inner',
            suffixes=('_odds','_perf')
        )

        matched = []
        for _, row in merged.iterrows():
            odds_row = {
                'Game Date': row['Game Date'],
                'Player Name': row['Player Name'],
                'Opponent': row.get('Opponent_odds', row.get('Opponent', None)),
                'Market': row.get('Market', ''),
                'O/U': row.get('O/U', np.nan),
                'American Over Odds': row.get('American Over Odds', np.nan),
                'American Under Odds': row.get('American Under Odds', np.nan),
                'Over Prob': row.get('Over Prob', np.nan),
                'Under Prob': row.get('Under Prob', np.nan),
                'Source': row.get('Source_odds', row.get('Source', None)),
            }
            perf_row = {
                'Date': row['Date'],
                'Player': row['Player'],
                'Opponent': row.get('Opponent_perf', row.get('Opponent', None)),
                'PTS': row.get('PTS', np.nan),
                'REB': row.get('REB', np.nan),
                'AST': row.get('AST', np.nan),
                'FG%': row.get('FG%', np.nan),
                'TO': row.get('TO', np.nan),
                'STL': row.get('STL', np.nan),
                'BLK': row.get('BLK', np.nan),
                'Source': row.get('Source_perf', row.get('Source', None)),
            }
            matched.append({'odds': odds_row, 'performance': perf_row})
        return matched

    # ---------- Projections CSV (optional) ----------

    def _normalize_projection_columns(self):
        df = self.projections_data
        rename_map = {}
        for base in ['PTS','REB','AST','TO','FG%']:
            for cand in [f'Proj_{base}', f'Projected {base}', f'{base}_Proj', f'Proj{base}']:
                if cand in df.columns:
                    rename_map[cand] = f'Proj_{base}'
                    break
        if 'Proj_STOCKS' not in df.columns:
            if 'Proj_STL' in df.columns and 'Proj_BLK' in df.columns:
                df['Proj_STOCKS'] = pd.to_numeric(df['Proj_STL'], errors='coerce') + pd.to_numeric(df['Proj_BLK'], errors='coerce')
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
        self.projections_data = df

    def _get_external_projection(self, player: str, date: pd.Timestamp, timeframe: str) -> Optional[Dict[str, float]]:
        if self.projections_data is None:
            return None
        df = self.projections_data
        mask = (
            df['Player'].astype(str).str.strip().str.lower() == str(player).strip().lower()
        ) & (pd.to_datetime(df['Date']) == pd.to_datetime(date))
        rows = df.loc[mask]
        if rows.empty:
            return None
        row = rows.iloc[0]
        out = {}
        tf_prefix = 'ProjWeekly' if timeframe == 'weekly' else ('ProjMonthly' if timeframe == 'monthly' else None)
        for stat in STATS:
            val = None
            if tf_prefix and f'{tf_prefix}_{stat}' in row.index and pd.notna(row[f'{tf_prefix}_{stat}']):
                val = float(row[f'{tf_prefix}_{stat}'])
            elif f'Proj_{stat}' in row.index and pd.notna(row[f'Proj_{stat}']):
                val = float(row[f'Proj_{stat}'])
            if val is not None:
                out[stat] = val
        return out if out else None

    # ---------- History / future windows ----------

    def _get_player_history(self, player: str, before_date: pd.Timestamp, n: int = 25) -> pd.DataFrame:
        df = self.performance_data
        hist = df[(df['Player'] == player) & (df['Date'] < before_date)].sort_values('Date')
        return hist.tail(n)

    def _get_player_season_to_date(self, player: str, before_date: pd.Timestamp) -> pd.DataFrame:
        return self._get_player_history(player, before_date, n=10_000)

    def _get_future_games(self, player: str, on_or_after_date: pd.Timestamp, n: int) -> pd.DataFrame:
        df = self.performance_data
        fut = df[(df['Player'] == player) & (df['Date'] >= on_or_after_date)].sort_values('Date')
        return fut.head(n)

    # ---------- SD multipliers/floors ----------

    def _rookie_sd_multiplier(self, games_played_before: int) -> Optional[float]:
        if games_played_before < 5:
            return None
        if games_played_before < 10:
            return 0.85
        if games_played_before < 25:
            return 0.95
        return 1.0

    def _timeframe_sd_multiplier(self, timeframe: str) -> float:
        if timeframe == 'weekly':
            return 0.6
        if timeframe == 'monthly':
            return 0.3
        return 1.0

    def _stat_specific_timeframe_multiplier(self, stat: str, timeframe: str) -> float:
        """Apply stat-specific timeframe multipliers for PTS, REB, AST only."""
        if stat in ['PTS', 'REB', 'AST']:
            if timeframe == 'intragame':
                return 0.75  # 75% of original
            elif timeframe == 'weekly':
                return 0.45  # 45% of original
            elif timeframe == 'monthly':
                return 0.25  # 25% of original
        # For other stats (STOCKS, TO, FG%), use the original timeframe multiplier
        return self._timeframe_sd_multiplier(timeframe)

    def _std_from_last_25(self, player: str, before_date: pd.Timestamp, stat: str, timeframe: str) -> Optional[float]:
        df_all = self.performance_data
        prior_all = df_all[(df_all['Player'] == player) & (df_all['Date'] < before_date)].sort_values('Date')
        games_played = len(prior_all)
        mult_rookie = self._rookie_sd_multiplier(games_played)
        if mult_rookie is None:
            return None

        hist = prior_all.tail(25)
        if len(hist) == 0:
            base = 0.02 if stat == 'FG%' else 0.2
        else:
            if stat == 'STOCKS':
                vals = (hist['STL'].astype(float) + hist['BLK'].astype(float))
            else:
                vals = hist[stat].astype(float)
            base = float(vals.std(ddof=1)) if len(vals) > 1 else 1e-6

        if stat == 'FG%':
            base = max(base, 0.02)
        else:
            base = max(base, 0.2)

        base *= mult_rookie
        base *= self._stat_specific_timeframe_multiplier(stat, timeframe)
        return float(base)

    # ---------- Odds helpers (μ = L − z·σ) ----------

    @staticmethod
    def _american_to_prob(american: float) -> Optional[float]:
        if pd.isna(american):
            return None
        a = float(american)
        if a > 0:
            return 100.0 / (a + 100.0)
        else:
            return (-a) / ((-a) + 100.0)

    @staticmethod
    def _renormalize(p_over: Optional[float], p_under: Optional[float]) -> Optional[float]:
        """Remove vig: return normalized over probability if both present; otherwise best guess."""
        if p_over is None and p_under is None:
            return None
        if p_over is not None and p_under is not None and p_over > 0 and p_under > 0:
            s = p_over + p_under
            if s > 0:
                return p_over / s
        # fallback: trust p_over if present; else infer from p_under
        if p_over is not None:
            return p_over
        return 1.0 - p_under if p_under is not None else None

    # Acklam's inverse normal CDF approximation
    @staticmethod
    def _inv_norm_cdf(p: float) -> float:
        """Return z such that Φ(z) = p. Uses a high-accuracy rational approximation."""
        # constants
        a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
              1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
        b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
              6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
        d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
              3.754408661907416e+00]
        plow  = 0.02425
        phigh = 1 - plow

        if p <= 0 or p >= 1 or not np.isfinite(p):
            # clamp to avoid NaN/inf
            p = min(max(p, 1e-12), 1 - 1e-12)

        if p < plow:
            q = math.sqrt(-2 * math.log(p))
            return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                   ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        elif p > phigh:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                     ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        else:
            q = p - 0.5
            r = q * q
            return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
                   (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

    def _odds_over_probability(self, odds_row: Dict) -> Optional[float]:
        """Return normalized Over probability from either pre-parsed probs or American odds."""
        # Prefer explicit probabilities if present
        p_over = odds_row.get('Over Prob')
        p_under = odds_row.get('Under Prob')

        if pd.notna(p_over) or pd.notna(p_under):
            p = self._renormalize(
                float(p_over) if pd.notna(p_over) else None,
                float(p_under) if pd.notna(p_under) else None
            )
            return p

        # Compute from American odds
        over_odds = odds_row.get('American Over Odds')
        under_odds = odds_row.get('American Under Odds')
        p_over0 = self._american_to_prob(over_odds)
        p_under0 = self._american_to_prob(under_odds)
        return self._renormalize(p_over0, p_under0)

    # ---------- Projections ----------

    def _project_intragame(self, player: str, game_date: pd.Timestamp, market_stat: str, market_line: float, odds_row: Dict) -> Dict[str, float]:
        """
        Intragame projection — now uses your μ = L - z·σ for PTS/REB/AST when we have a book line + odds.

        For non-market stats:
          FG%, TO, STOCKS = last 5 avg
        For the two of (PTS,REB,AST) that we don't have a market for on that row:
          use season-to-date averages (pre-game)
        """
        # external projections?
        ext = self._get_external_projection(player, game_date, 'intragame')
        if ext:
            return ext

        last5 = self._get_player_history(player, game_date, n=5)
        def mean_last5(col, default):
            return float(last5[col].mean()) if col in last5 and len(last5) else default

        proj = {
            'FG%': mean_last5('FG%', 0.45),
            'TO': mean_last5('TO', 2.0),
            'STOCKS': mean_last5('STL', 1.0) + mean_last5('BLK', 0.9),
        }

        season = self._get_player_season_to_date(player, game_date)
        def season_avg(col, default):
            return float(season[col].mean()) if col in season and len(season) else default

        # Which stat has a line?
        m = (market_stat or '').upper()
        stat_from_market = None
        if 'PTS' in m or 'POINT' in m:
            stat_from_market = 'PTS'
        elif 'REB' in m:
            stat_from_market = 'REB'
        elif 'AST' in m:
            stat_from_market = 'AST'

        # Compute μ = L - z·σ if possible
        if stat_from_market is not None and market_line is not None and not pd.isna(market_line):
            # σ from last-25 (intragame timeframe multiplier = 1.0)
            sigma = self._std_from_last_25(player, game_date, stat_from_market, 'intragame')
            p_over = self._odds_over_probability(odds_row)
            mu = None
            if sigma is not None and sigma > 0 and p_over is not None and 0.0 < p_over < 1.0:
                # If L is the level where P(X > L) = p_over, then P(X <= L) = 1 - p_over
                zline = self._inv_norm_cdf(1.0 - float(p_over))
                mu = float(market_line) - zline * float(sigma)
            else:
                # Fallback: use the line directly if we can't compute z
                mu = float(market_line)

            proj[stat_from_market] = mu

            # Fill the other two with season averages
            for s in ('PTS','REB','AST'):
                if s != stat_from_market:
                    proj[s] = season_avg(s, {'PTS':20.0, 'REB':6.0, 'AST':4.0}[s])
        else:
            # No useful market on this row — fallback to season avgs for PTS/REB/AST
            proj['PTS'] = season_avg('PTS', 20.0)
            proj['REB'] = season_avg('REB', 6.0)
            proj['AST'] = season_avg('AST', 4.0)

        return proj

    def _project_blend(self, player: str, game_date: pd.Timestamp, recent_n: int, timeframe: str) -> Dict[str, float]:
        # Prefer external projections for weekly/monthly if present
        ext = self._get_external_projection(player, game_date, timeframe)
        if ext and all(k in ext for k in STATS):
            return ext

        season = self._get_player_season_to_date(player, game_date)
        recent = season.tail(recent_n)

        def mean_or0(df, col):
            return float(df[col].mean()) if col in df and len(df) else 0.0

        proj = {}
        for stat in ['PTS','REB','AST','FG%','TO']:
            proj[stat] = 0.6*mean_or0(season, stat) + 0.4*mean_or0(recent, stat)
        proj['STOCKS'] = 0.6*(mean_or0(season,'STL')+mean_or0(season,'BLK')) + 0.4*(mean_or0(recent,'STL')+mean_or0(recent,'BLK'))
        return proj

    # ---------- PPS (z-scores) ----------

    def _compute_zscores(self, player: str, date: pd.Timestamp, projection: Dict[str,float], actual: Dict[str,float], timeframe: str) -> Optional[Dict[str,float]]:
        zs = {}
        for stat in STATS:
            if (stat not in projection) or (stat not in actual) or pd.isna(actual[stat]):
                continue
            std = self._std_from_last_25(player, date, stat, timeframe)
            if std is None:
                return None
            if stat == 'TO':
                zs[stat] = (projection[stat] - actual[stat]) / std  # fewer TO better
            else:
                zs[stat] = (actual[stat] - projection[stat]) / std
        return zs

    # ---------- Simulation ----------

    def simulate_trade(self,
                       projection: Dict[str,float],
                       actual: Dict[str,float],
                       old_price: float,
                       position: str,
                       strategy: str,
                       timeframe: str,
                       player: str,
                       date: pd.Timestamp,
                       opponent: str) -> Optional[Dict]:
        z_scores = self._compute_zscores(player, date, projection, actual, timeframe)
        if z_scores is None:
            return None

        # Weighted PPS (your weights are handled elsewhere in your project; if you keep them here, add accordingly)
        # For clarity, we compute PPS directly from z_scores with the same weights you’ve specified in earlier steps:
        weights = {'PTS':0.475,'AST':0.15,'REB':0.15,'STOCKS':0.075,'FG%':0.10,'TO':0.05}
        raw_pps = 0.0
        for k, w in weights.items():
            raw_pps += w * float(z_scores.get(k, 0.0))

        damp = PriceDampening.dampen_delta(raw_pps, strategy, timeframe)
        new_price = PriceDampening.calculate_final_price(old_price, damp)
        price_change_pct = ((new_price - old_price) / old_price) * 100.0

        bank_pnl_pct = -price_change_pct if position == 'long' else price_change_pct
        bank_pnl_dollars = float(self.stake_per_trade) * (bank_pnl_pct / 100.0)
        user_pnl_dollars = -bank_pnl_dollars

        return {
            'strategy': strategy,
            'timeframe': timeframe,
            'position': position,
            'player': player,
            'date': date,
            'opponent': opponent,
            'old_price': old_price,
            'new_price': new_price,
            'raw_pps': raw_pps,
            'dampened_delta': damp,
            'z_scores': z_scores,
            'price_change_pct': price_change_pct,
            'bank_pnl_pct': bank_pnl_pct,
            'bank_pnl_dollars': bank_pnl_dollars,
            'user_pnl_dollars': user_pnl_dollars,
        }

    # ---------- Entry ----------

    def run_backtest(self, strategy: str, timeframe: str, position: str) -> pd.DataFrame:
        if strategy not in ('conservative','aggressive'):
            raise ValueError("strategy must be 'conservative' or 'aggressive'")
        if timeframe not in ('intragame','weekly','monthly'):
            raise ValueError("timeframe must be 'intragame','weekly','monthly'")
        if position not in ('long','short','both'):
            raise ValueError("position must be 'long','short', or 'both'")

        print(f"Attempting to match {len(self.odds_data)} odds records with {len(self.performance_data)} performance records...")
        matched = self.match_games()
        print(f"Successfully matched {len(matched)} games")

        results = []
        for mg in matched:
            odds = mg['odds']
            perf = mg['performance']
            player = odds['Player Name']
            date = pd.to_datetime(odds['Game Date'])
            opponent = odds.get('Opponent') or perf.get('Opponent')

            if timeframe == 'intragame':
                proj = self._project_intragame(player, date, odds.get('Market',''), odds.get('O/U', np.nan), odds)
                actual = {
                    'PTS': float(perf['PTS']),
                    'REB': float(perf['REB']),
                    'AST': float(perf['AST']),
                    'FG%': float(perf['FG%']),
                    'TO':  float(perf['TO']),
                    'STOCKS': float(perf['STL']) + float(perf['BLK']),
                }
            elif timeframe == 'weekly':
                proj = self._project_blend(player, date, recent_n=3, timeframe='weekly')
                fut = self._get_future_games(player, date, n=3)
                if len(fut) < 3:
                    continue
                actual = {
                    'PTS': float(fut['PTS'].mean()),
                    'REB': float(fut['REB'].mean()),
                    'AST': float(fut['AST'].mean()),
                    'FG%': float(fut['FG%'].mean()),
                    'TO':  float(fut['TO'].mean()),
                    'STOCKS': float((fut['STL'] + fut['BLK']).mean()),
                }
            else:
                proj = self._project_blend(player, date, recent_n=12, timeframe='monthly')
                fut = self._get_future_games(player, date, n=12)
                if len(fut) < 12:
                    continue
                actual = {
                    'PTS': float(fut['PTS'].mean()),
                    'REB': float(fut['REB'].mean()),
                    'AST': float(fut['AST'].mean()),
                    'FG%': float(fut['FG%'].mean()),
                    'TO':  float(fut['TO'].mean()),
                    'STOCKS': float((fut['STL'] + fut['BLK']).mean()),
                }

            positions = ['long','short'] if position == 'both' else [position]
            for pos in positions:
                trade = self.simulate_trade(
                    proj, actual, 25.0, pos, strategy, timeframe,
                    player, date, opponent
                )
                if trade is not None:
                    results.append(trade)

        if not results:
            return pd.DataFrame(columns=[
                'strategy','timeframe','position','player','date','opponent',
                'old_price','new_price','raw_pps','dampened_delta','z_scores',
                'price_change_pct','bank_pnl_pct','bank_pnl_dollars','user_pnl_dollars'
            ])

        df = pd.DataFrame(results)
        print(f"Total trades: {len(df)}")
        win_rate = (df['bank_pnl_pct'] > 0).mean() * 100.0
        print(f"Bank win rate: {win_rate:.1f}%")
        avg_pct = df['bank_pnl_pct'].mean()
        print(f"Average bank P&L (pct): {avg_pct:.2f}%")
        avg_usd = df['bank_pnl_dollars'].mean()
        print(f"Average bank P&L ($/trade @stake={self.stake_per_trade:.0f}): {avg_usd:.2f}")
        total_bank = df['bank_pnl_dollars'].sum()
        print(f"Total bank P&L ($): {total_bank:.2f}")
        return df
