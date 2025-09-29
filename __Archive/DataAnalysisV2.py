#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SeasonAnalyzer — player-level impacts with robust missing-game handling and DB→Excel/CSV export
"""
import os
# --- unify threading & OpenMP runtime to avoid double-free ---
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")     # <-- key when libgomp is present
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")       # stabilize forking in kernels
# Optional: temporary diagnostic escape hatch (remove once stable)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


# import sys
import traceback
import time
# import pandas as pd

import json, sqlite3
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Lasso

# After all scientific imports:
from threadpoolctl import threadpool_limits, threadpool_info
# Force single-thread compute in all linked BLAS/OpenMP backends
threadpool_limits(1)
try:
    import mkl
    mkl.set_num_threads(1)  # belt-and-suspenders for MKL
except Exception:
    pass
try:
    print("[Threadpools]", ", ".join(f"{b.get('internal_api','?')}:{b.get('num_threads','?')}" for b in threadpool_info()))
except Exception:
    pass

try:
    backends = threadpool_info()
    print("[Threadpools]", ", ".join(f"{b.get('internal_api','?')}:{b.get('num_threads','?')}" for b in backends))
except Exception:
    pass


class SeasonAnalyzer:
    def __init__(self, league_name="Bundesliga", country_name="Germany", season=2023, lasso_alphas=[0.01, 0.001]):
        # Config
        self.YearOfDownload = 2025
        self.AgeReduction = 2025 - season
        self.LEAGUE_NAME = league_name
        self.COUNTRY_NAME = country_name
        self.SEASON = season
        self.SAVE_DIR = f"{country_name}_{league_name}/{season}"
        self.FIXTURES_FILE = "fixtures.json"
        self.EVENTS_FILE = "match_events.json"
        self.PLAYER_FILE = "players.json"
        self.lasso_alphas = list(lasso_alphas) if isinstance(lasso_alphas, (list, tuple, set)) else [lasso_alphas]

        self.minutes_equiv_tolerance = 0  # set to 1 (or more) if you want near-equality instead of exact minutes

        os.makedirs(self.SAVE_DIR, exist_ok=True)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.DB_PATH = os.path.join(script_dir, "analysis_results.db")

        # Data containers
        self.fixtures_data = None
        self.match_events = None
        self.players = None

        self.team_players = {}        # {team_id: set(player_ids)}
        self.player_id_to_name = {}   # {player_id: player_name}
        self.player_id_to_position = {}
        self.player_id_to_age = {}    # {player_id: player_age (season-adjusted)}

        self.segment_df = None        # per-segment design, player columns p<id>
        self.match_summary_df = None
        self.team_summary_df = None
        self.coef_df_final = None
        self.player_game_metrics = {}

        # Missing-data accounting
        self.total_regular_games = 0
        self.valid_regular_games = 0
        self.missing_games_count = 0
        self.valid_fixture_ids = set()

    # ---------- IO ----------
    def load_data(self):
        fixtures_path = os.path.join(self.SAVE_DIR, self.FIXTURES_FILE)
        events_path   = os.path.join(self.SAVE_DIR, self.EVENTS_FILE)
        player_path   = os.path.join(self.SAVE_DIR, self.PLAYER_FILE)

        with open(fixtures_path, 'r', encoding='utf-8') as f:
            self.fixtures_data = json.load(f)['response']
        with open(events_path, 'r', encoding='utf-8') as f:
            self.match_events = json.load(f)
        with open(player_path, 'r', encoding='utf-8') as f:
            self.players = json.load(f)

        # Ages mapped & adjusted to season
        for _, roster in self.players.items():
            for px in roster:
                pid = px.get('id')
                age = px.get('age')
                if pid is not None and age is not None:
                    self.player_id_to_age[pid] = age - self.AgeReduction

    # ---------- helpers ----------
    def _get_minute(self, ev):
        extra = ev.get('time', {}).get('extra') or 0
        return (ev.get('time', {}).get('elapsed') or 0) + extra

    def _is_regular(self, fix):
        round_str = fix.get('league', {}).get('round', '') or ''
        return round_str.startswith('Regular Season')

    def _fixture_has_required_data(self, fid: str) -> bool:
        ev = self.match_events.get(fid, {})
        return bool(ev.get('lineups')) and bool(ev.get('events'))

    # ---------- main builders ----------
    def identify_players_and_build_segments(self):
        """
        Build player presence segments using ONLY fixtures with both lineups and events.
        Count & remove missing games from analysis.
        """
        # Count total regular-season fixtures and determine validity
        reg_fixture_ids = []
        for fix in self.fixtures_data:
            if self._is_regular(fix):
                reg_fixture_ids.append(str(fix['fixture']['id']))
        self.total_regular_games = len(reg_fixture_ids)

        self.valid_fixture_ids = {fid for fid in reg_fixture_ids if self._fixture_has_required_data(fid)}
        self.valid_regular_games = len(self.valid_fixture_ids)
        self.missing_games_count = self.total_regular_games - self.valid_regular_games

        # Track rosters / positions
        for fix in self.fixtures_data:
            fid = str(fix['fixture']['id'])
            if fid not in self.valid_fixture_ids:
                continue
            evpack = self.match_events[fid]
            # starting lineups
            for lineup in evpack.get('lineups', []):
                team_id = lineup['team']['id']
                self.team_players.setdefault(team_id, set())
                for entry in lineup.get('startXI', []):
                    pid = entry['player']['id']
                    pname = entry['player']['name']
                    ppos = entry['player'].get('pos', entry['player'].get('position', 'Unknown'))
                    self.team_players[team_id].add(pid)
                    self.player_id_to_name[pid] = pname
                    self.player_id_to_position[pid] = ppos
            # subs (both in/out)
            for ev in evpack.get('events', []):
                if ev.get('type') == 'subst':
                    team_id = ev['team']['id']
                    self.team_players.setdefault(team_id, set())
                    pid_out = ev.get('player', {}).get('id')
                    pname_out = ev.get('player', {}).get('name')
                    pid_in = ev.get('assist', {}).get('id')
                    pname_in = ev.get('assist', {}).get('name')
                    if pid_in:
                        self.team_players[team_id].add(pid_in)
                        self.player_id_to_name[pid_in] = pname_in
                    if pid_out:
                        self.team_players[team_id].add(pid_out)
                        self.player_id_to_name[pid_out] = pname_out

        # Segment matrix columns (individual players only; no grouping)
        all_player_ids = sorted({pid for s in self.team_players.values() for pid in s})
        player_cols = [f"p{pid}" for pid in all_player_ids]
        self.segment_df = pd.DataFrame(columns=player_cols)

        # Build segments per valid match
        meta_rows = []
        for fix in self.fixtures_data:
            fid = str(fix['fixture']['id'])
            if fid not in self.valid_fixture_ids:
                continue

            evpack = self.match_events[fid]
            home_team = fix['teams']['home']['name']
            away_team = fix['teams']['away']['name']
            home_id   = fix['teams']['home']['id']
            away_id   = fix['teams']['away']['id']

            match_events = evpack.get('events', [])
            match_end_min = max([self._get_minute(e) for e in match_events] + [90])

            # initial on-field set
            current = {}
            for lineup in evpack.get('lineups', []):
                team_sign = 1 if lineup['team']['id'] == home_id else -1
                for entry in lineup.get('startXI', []):
                    current[entry['player']['id']] = team_sign

            subs = sorted([e for e in match_events if e.get('type') == 'subst'], key=self._get_minute)
            segment_start = 0
            boundaries = sorted({self._get_minute(e) for e in subs} | {match_end_min})

            for seg_end in boundaries:
                duration = seg_end - segment_start
                # goals in segment
                home_goals = sum(1 for e in match_events
                                 if e.get('type') == 'Goal' and e['team']['id'] == home_id
                                 and segment_start < self._get_minute(e) <= seg_end)
                away_goals = sum(1 for e in match_events
                                 if e.get('type') == 'Goal' and e['team']['id'] == away_id
                                 and segment_start < self._get_minute(e) <= seg_end)
                goal_diff = home_goals - away_goals

                row = {f"p{pid}": 0 for pid in current.keys()}
                for pid, sign in current.items():
                    row[f"p{pid}"] = sign

                self.segment_df = pd.concat([self.segment_df, pd.DataFrame([row])], ignore_index=True)
                meta_rows.append({
                    'fixture': fid,
                    'home_team': home_team, 'away_team': away_team,
                    'home_id': home_id, 'away_id': away_id,
                    'start': segment_start, 'end': seg_end,
                    'duration': duration, 'goal_diff': goal_diff
                })

                # apply subs at boundary
                for e in [x for x in subs if self._get_minute(x) == seg_end]:
                    pid_out = e.get('player', {}).get('id')
                    pid_in  = e.get('assist', {}).get('id')
                    if pid_out in current:
                        current.pop(pid_out, None)
                    if pid_in:
                        current[pid_in] = (1 if e['team']['id'] == home_id else -1)

                segment_start = seg_end

        meta_df = pd.DataFrame(meta_rows)
        self.segment_df = pd.concat([meta_df, self.segment_df], axis=1)

    def calculate_game_metrics(self):
        """Per-player appearances and minutes, using only valid fixtures."""
        self.player_game_metrics = {pid: {'games_started': 0, 'full_games_played': 0,
                                          'games_subbed_on': 0, 'games_subbed_off': 0,
                                          'minutes_played': 0}
                                    for pid in self.player_id_to_name}

        for fix in self.fixtures_data:
            fid = str(fix['fixture']['id'])
            if fid not in self.valid_fixture_ids:
                continue
            evpack = self.match_events[fid]
            events = evpack.get('events', [])
            starters = set()
            subs_out = set()
            match_end_min = max([self._get_minute(e) for e in events] + [90])

            for lineup in evpack.get('lineups', []):
                for px in lineup.get('startXI', []):
                    pid = px['player']['id']
                    starters.add(pid)
                    self.player_game_metrics[pid]['games_started'] += 1

            for e in events:
                if e.get('type') == 'subst':
                    pid_out = e.get('player', {}).get('id')
                    pid_in  = e.get('assist', {}).get('id')
                    minute  = self._get_minute(e)
                    if pid_out:
                        subs_out.add(pid_out)
                        self.player_game_metrics[pid_out]['games_subbed_off'] += 1
                        self.player_game_metrics[pid_out]['minutes_played'] += minute
                    if pid_in:
                        self.player_game_metrics[pid_in]['games_subbed_on'] += 1
                        self.player_game_metrics[pid_in]['minutes_played'] += (match_end_min - minute)

            for pid in (starters - subs_out):
                self.player_game_metrics[pid]['full_games_played'] += 1
                self.player_game_metrics[pid]['minutes_played'] += match_end_min

    def summarize_matches_and_teams(self):
        """Match summary and team table (valid fixtures only)."""
        match_summary_list = []
        for fix in self.fixtures_data:
            fid = str(fix['fixture']['id'])
            if fid not in self.valid_fixture_ids:
                continue
            evs = self.match_events[fid].get('events', [])
            home_team = fix['teams']['home']['name']
            away_team = fix['teams']['away']['name']
            home_id   = fix['teams']['home']['id']
            away_id   = fix['teams']['away']['id']
            # matchday
            round_str = fix.get('league', {}).get('round', '')
            try:
                matchday = int((round_str.split('-')[-1]).strip())
            except:
                matchday = None
            # unchanged starters
            start_players = {px['player']['id'] for lu in self.match_events[fid].get('lineups', [])
                             for px in lu.get('startXI', [])}
            subs_out_players = {e.get('player', {}).get('id') for e in evs if e.get('type') == 'subst'}
            unchanged = len(start_players - subs_out_players)
            # subs by minute
            subs_by_min = {}
            for e in evs:
                if e.get('type') == 'subst':
                    minute = self._get_minute(e)
                    subs_by_min.setdefault((e['team']['id'], minute), []).append(e)
            total_subs = len(subs_by_min)
            # goals
            home_goals = sum(1 for e in evs if e.get('type') == 'Goal' and e['team']['id'] == home_id)
            away_goals = sum(1 for e in evs if e.get('type') == 'Goal' and e['team']['id'] == away_id)

            match_summary_list.append({
                'matchday': matchday, 'fixture_id': fid,
                'home_team': home_team, 'away_team': away_team,
                'home_team_id': home_id, 'away_team_id': away_id,
                'home_goals': home_goals, 'away_goals': away_goals,
                'sub_events': total_subs, 'unchanged_players': unchanged
            })
        self.match_summary_df = pd.DataFrame(match_summary_list)

        # Team table
        team_summary_list = []
        total_minutes = self.segment_df['duration'].sum() if not self.segment_df.empty else 0
        for team_id, players in self.team_players.items():
            team_matches = self.match_summary_df[
                (self.match_summary_df['home_team_id'] == team_id) |
                (self.match_summary_df['away_team_id'] == team_id)
            ]
            wins = ((team_matches['home_team_id'] == team_id) & (team_matches['home_goals'] > team_matches['away_goals'])).sum() \
                 + ((team_matches['away_team_id'] == team_id) & (team_matches['away_goals'] > team_matches['home_goals'])).sum()
            draws = (team_matches['home_goals'] == team_matches['away_goals']).sum()
            losses = len(team_matches) - wins - draws
            goals_scored = ((team_matches['home_team_id'] == team_id) * team_matches['home_goals'] +
                            (team_matches['away_team_id'] == team_id) * team_matches['away_goals']).sum()
            goals_conceded = ((team_matches['home_team_id'] == team_id) * team_matches['away_goals'] +
                              (team_matches['away_team_id'] == team_id) * team_matches['home_goals']).sum()
            always_playing = sum(self.segment_df.get(f"p{pid}", pd.Series(dtype=float)).abs().sum() == total_minutes
                                 for pid in players)

            team_summary_list.append({
                'team_id': team_id, 'wins': wins, 'draws': draws, 'losses': losses,
                'goals_scored': goals_scored, 'goals_conceded': goals_conceded,
                'total_players_used': len(players), 'always_playing': always_playing
            })

        self.team_summary_df = pd.DataFrame(team_summary_list)
        self.team_summary_df['points'] = self.team_summary_df['wins']*3 + self.team_summary_df['draws']
        self.team_summary_df.sort_values(by=['points','goals_scored'], ascending=[False, False], inplace=True)
        self.team_summary_df.reset_index(drop=True, inplace=True)
        self.team_summary_df['league_position'] = self.team_summary_df.index + 1

    def compute_impacts(self):
        """
        Player-level impacts: WLS plus-minus and Lasso.
        No grouping; add minutes-equivalence column.
        """
        y = pd.to_numeric(self.segment_df['goal_diff'], errors='coerce').fillna(0)
        w = pd.to_numeric(self.segment_df['duration'], errors='coerce').fillna(1)

        exclude = {'fixture','home_team','away_team','home_id','away_id','start','end','duration','goal_diff'}
        X_df = self.segment_df.drop(columns=[c for c in exclude if c in self.segment_df], errors='ignore')
        X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)


        def fit_wls(X, y, w):
            Xc = sm.add_constant(X, has_constant='add')
            yv = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
            wv = np.ascontiguousarray(np.asarray(w, dtype=np.float64))
            Xv = np.ascontiguousarray(np.asarray(Xc, dtype=np.float64))
            ok = np.isfinite(yv) & np.isfinite(wv) & np.isfinite(Xv).all(axis=1)
            if not np.any(ok):
                raise ValueError("No valid rows for WLS after filtering.")
            with threadpool_limits(1):
                model = sm.WLS(yv[ok], Xv[ok, :], weights=wv[ok]).fit(method="pinv")  # <-- use pinv solver
            params = pd.Series(model.params, index=Xc.columns)
            bse = pd.Series(model.bse, index=Xc.columns)
            return params.drop('const', errors='ignore'), bse.drop('const', errors='ignore') 

        coef, err = fit_wls(X_df, y, w)

        # assemble per-player DataFrame
        coef_df = pd.DataFrame({
            'player_id': coef.index,
            'impact': coef.values,
            'impact_std': err.reindex(coef.index).values
        })

        # minutes played (exact from segments)
        presence = X_df.abs()
        minutes_played = presence.mul(self.segment_df['duration'], axis=0).sum()
        coef_df['minutes_played'] = coef_df['player_id'].map(minutes_played.to_dict()).fillna(0).astype(float)

        # names, team(s), positions
        def name_from_pid(pid_str):
            if pid_str.startswith('p'):
                return self.player_id_to_name.get(int(pid_str[1:]), 'Unknown')
            return pid_str

        def teams_for(pid_str):
            col = pid_str
            if col not in self.segment_df:
                return None
            homes = self.segment_df.loc[self.segment_df[col] == 1, 'home_team'].dropna().unique().tolist()
            aways = self.segment_df.loc[self.segment_df[col] == -1, 'away_team'].dropna().unique().tolist()
            teams = sorted(set(homes + aways))
            if len(teams) == 1:
                return teams[0]
            elif len(teams) > 1:
                return f"Multiple Teams ({', '.join(teams)})"
            return None

        def position_from_pid(pid_str):
            if pid_str.startswith('p'):
                return self.player_id_to_position.get(int(pid_str[1:]), 'Unknown')
            return 'Unknown'

        coef_df['player_name'] = coef_df['player_id'].apply(name_from_pid)
        coef_df['team(s)'] = coef_df['player_id'].apply(teams_for)
        coef_df['position'] = coef_df['player_id'].apply(position_from_pid)

        # league positions (map by team_summary_df)
        id_to_name = {}
        for fix in self.fixtures_data:
            id_to_name[fix['teams']['home']['id']] = fix['teams']['home']['name']
            id_to_name[fix['teams']['away']['id']] = fix['teams']['away']['name']
        team_position_map = {}
        if self.team_summary_df is not None:
            for _, row in self.team_summary_df.iterrows():
                tid = row['team_id']; pos = row['league_position']
                if tid in id_to_name:
                    team_position_map[id_to_name[tid]] = pos

        def league_pos(team_str):
            if not team_str: return None
            if isinstance(team_str, str) and team_str.startswith('Multiple Teams'):
                return 'Multiple Teams'
            return team_position_map.get(team_str, None)

        coef_df['league_position'] = coef_df['team(s)'].apply(league_pos)

        # Lasso
        valid = np.isfinite(y) & np.isfinite(X_df).all(axis=1)
        X_lasso = X_df[valid]; y_lasso = y[valid]
        for alpha in self.lasso_alphas:
            
            a = float(alpha); a_str = str(a).replace('.', '_')
            with threadpool_limits(1):   # <-- add
                model = Lasso(alpha=a, fit_intercept=True)
                model.fit(X_lasso, y_lasso)
            series = pd.Series(model.coef_, index=X_lasso.columns)
            coef_df[f'lasso_impact_alpha_{a_str}'] = coef_df['player_id'].map(series.to_dict())
            coef_df[f'lasso_std_alpha_{a_str}'] = float(series.std())

        # per-player game metrics
        def metric(pid_str, key):
            pid = int(pid_str[1:]) if pid_str.startswith('p') else None
            return self.player_game_metrics.get(pid, {}).get(key, 0) if pid else 0

        for m in ['games_started','full_games_played','games_subbed_on','games_subbed_off']:
            coef_df[m] = coef_df['player_id'].apply(lambda pid_str, k=m: metric(pid_str, k))

        coef_df['FTE_games_played'] = (coef_df['minutes_played'] / 90).round(2)

        # Add age
        def age_for(pid_str):
            if pid_str.startswith('p'):
                return self.player_id_to_age.get(int(pid_str[1:]), None)
            return None
        coef_df['age'] = coef_df['player_id'].apply(age_for)

        # ---- NEW: players with IDENTICAL on-field timeline (exact same segments & signs) ----
        # Build a signature per player column from the presence matrix (values in {-1,0,1})
        presence_mat = X_df.astype('int8')  # ensure stable, compact dtype
        
        # group columns by identical vectors
        sig_to_cols = {}
        for col in presence_mat.columns:
            sig = tuple(presence_mat[col].tolist())
            sig_to_cols.setdefault(sig, []).append(col)
        
        def identical_peers(pid_str):
            # pid_str is like 'p12345'
            col = pid_str
            if col not in presence_mat.columns:
                return ""
            sig = tuple(presence_mat[col].tolist())
            peers = [c for c in sig_to_cols.get(sig, []) if c != col]
            # Return names; fall back to IDs if unknown
            return ", ".join(self.player_id_to_name.get(int(c[1:]), 'Unknown') for c in peers)
        
        coef_df['other_players_same_minutes'] = coef_df['player_id'].apply(identical_peers)

        # Constant context + missing-games stats
        coef_df['country'] = self.COUNTRY_NAME
        coef_df['league']  = self.LEAGUE_NAME
        coef_df['season']  = self.SEASON
        coef_df['total_regular_games'] = self.total_regular_games
        coef_df['valid_regular_games'] = self.valid_regular_games
        coef_df['missing_games_count'] = self.missing_games_count

        # Final sort & stash
        coef_df.sort_values(by='impact', ascending=False, inplace=True)
        coef_df.reset_index(drop=True, inplace=True)
        self.coef_df_final = coef_df

    def add_goals_assists(self):
        """Totals per individual player; valid fixtures only."""
        goals = {}; assists = {}
        for fix in self.fixtures_data:
            fid = str(fix['fixture']['id'])
            if fid not in self.valid_fixture_ids:
                continue
            for ev in self.match_events[fid].get('events', []):
                if ev.get('type') == 'Goal':
                    detail = ev.get('detail', '')
                    if isinstance(detail, str) and 'Own Goal' in detail:
                        continue
                    scorer = ev.get('player', {}).get('id')
                    helper = ev.get('assist', {}).get('id')
                    if scorer: goals[scorer] = goals.get(scorer, 0) + 1
                    if helper: assists[helper] = assists.get(helper, 0) + 1

        def g_for(pid_str):
            pid = int(pid_str[1:]) if pid_str.startswith('p') else None
            return goals.get(pid, 0) if pid else 0
        def a_for(pid_str):
            pid = int(pid_str[1:]) if pid_str.startswith('p') else None
            return assists.get(pid, 0) if pid else 0

        self.coef_df_final['goals'] = self.coef_df_final['player_id'].apply(g_for)
        self.coef_df_final['assists'] = self.coef_df_final['player_id'].apply(a_for)

    # ---------- Pipeline ----------
    def run_analysis(self):
        self.identify_players_and_build_segments()
        self.calculate_game_metrics()
        self.summarize_matches_and_teams()
        self.compute_impacts()
        self.add_goals_assists()

    # ---------- Persistence ----------
    def save_results(self, output_path=None):
        """Save to Excel (xlsx), append/replace in SQLite, then read back & export DB → xlsx/csv."""
        if output_path is None:
            output_path = os.path.join(self.SAVE_DIR, "output.xlsx")

        # Excel (OLS sheet + Lasso sheets)
        ols_df = self.coef_df_final.sort_values(by='impact', ascending=False)
        ols_df.to_excel(output_path, sheet_name="OLS", index=False)
        with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            for alpha in self.lasso_alphas:
                a = float(alpha); a_str = str(a).replace('.', '_')
                col = f'lasso_impact_alpha_{a_str}'
                if col not in self.coef_df_final.columns:
                    continue
                df_ = self.coef_df_final[self.coef_df_final[col].fillna(0) != 0].copy()
                df_.sort_values(by=col, ascending=False, inplace=True)
                df_.to_excel(writer, sheet_name=f"Lasso_{a_str}", index=False)

        # SQLite write (schema-stable)
        conn = sqlite3.connect(self.DB_PATH)
        cur = conn.cursor()
        exists = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_results';"
        ).fetchone() is not None

        if exists:
            existing_cols = {c[1] for c in cur.execute("PRAGMA table_info(analysis_results);").fetchall()}
            new_cols = set(self.coef_df_final.columns)
            if existing_cols != new_cols:
                # create versioned table to avoid hard failure
                self.coef_df_final.to_sql(name="analysis_results_new", con=conn, if_exists="replace", index=False)
                conn.commit()
            else:
                cur.execute(
                    "DELETE FROM analysis_results WHERE country=? AND league=? AND season=?",
                    (self.COUNTRY_NAME, self.LEAGUE_NAME, self.SEASON)
                )
                conn.commit()
                self.coef_df_final.to_sql(name="analysis_results", con=conn, if_exists="append", index=False)
        else:
            self.coef_df_final.to_sql(name="analysis_results", con=conn, if_exists="replace", index=False)
        conn.close()

        # ---- NEW: Read back from DB and export to XLSX + CSV ----
        conn = sqlite3.connect(self.DB_PATH)
        query = """
            SELECT *
            FROM analysis_results
            WHERE country = ? AND league = ? AND season = ?
        """
        df_from_db = pd.read_sql_query(query, conn, params=[self.COUNTRY_NAME, self.LEAGUE_NAME, self.SEASON])
        conn.close()

        out_xlsx = os.path.join(self.SAVE_DIR, "analysis_from_db.xlsx")
        out_csv  = os.path.join(self.SAVE_DIR, "analysis_from_db.csv")
        df_from_db.to_excel(out_xlsx, index=False)
        df_from_db.to_csv(out_csv, index=False)

    def run(self):
        self.load_data()
        self.run_analysis()
        self.save_results()
        

        
class SeasonBatchRunner:
    """
    Reads `downloads_progress.xlsx`, selects completed downloads, and runs SeasonAnalyzer per row.

    - Flexible filter: by default requires all *_done flags true. You can switch to "events_only".
    - Skips rows where the expected JSON inputs are missing or malformed (preflight).
    - Spawns a subprocess per season to survive native crashes (double-free, OMP conflicts).
    - Emits a batch report CSV/XLSX with success/fail and timing.
    """

    def __init__(
        self,
        progress_filename: str = "downloads_progress.xlsx",
        filter_mode: str = "all_done",  # "all_done" or "events_only"
        minutes_equiv_tolerance: int = 0,  # pass-through to analyzer if desired
        lasso_alphas=(0.01, 0.001),
        debug: bool = False,
    ):
        import os, pandas as pd

        self.debug = debug
        self.filter_mode = filter_mode
        self.lasso_alphas = list(lasso_alphas)
        self.minutes_equiv_tolerance = minutes_equiv_tolerance

        # Resolve paths relative to this script
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.script_path = os.path.abspath(__file__)
        self.progress_path = os.path.join(self.script_dir, progress_filename)

        # Load progress with a small schema helper
        if not os.path.exists(self.progress_path):
            raise FileNotFoundError(f"Could not find {self.progress_path}")

        self.progress_df = pd.read_excel(self.progress_path)

        # Normalize columns (lowercase)
        self.progress_df.columns = [str(c).strip().lower() for c in self.progress_df.columns]

        # Heuristic mapping
        self.col_country = self._first_of(("country",))
        self.col_league  = self._first_of(("league",))
        self.col_season  = self._first_of(("season",))

        # Done flags (best-effort detection)
        self.col_teams_done    = self._first_of(("teams_done",))
        self.col_players_done  = self._first_of(("players_done", "players_status_done"))
        self.col_fixtures_done = self._first_of(("fixtures_done",))
        self.col_events_done   = self._first_of(("events_done",))

        missing_core = [cname for cname in [self.col_country, self.col_league, self.col_season] if cname is None]
        if missing_core:
            raise ValueError(f"Progress file missing required columns: {missing_core}")

    def _first_of(self, candidates):
        for c in candidates:
            if c in self.progress_df.columns:
                return c
        return None

    def _row_is_complete(self, row) -> bool:
        """
        Decide whether a competition is "done" and ready for analysis.
        filter_mode:
          - "all_done": all available *_done flags must be True
          - "events_only": only events_done must be True (lightest requirement)
        """
        import pandas as pd

        def as_bool(val):
            if isinstance(val, bool): return val
            if pd.isna(val): return False
            if isinstance(val, (int, float)): return val != 0
            return str(val).strip().lower() in ("true", "t", "yes", "y", "1")

        if self.filter_mode == "events_only" and self.col_events_done:
            return as_bool(row.get(self.col_events_done, False))

        # all_done: gather every *_done column that exists, require all True
        flags = []
        for col in (self.col_teams_done, self.col_players_done, self.col_fixtures_done, self.col_events_done):
            if col is not None:
                flags.append(as_bool(row.get(col, False)))
        return bool(flags) and all(flags)

    def _json_inputs_exist(self, country, league, season) -> bool:
        import os
        save_dir = os.path.join(self.script_dir, f"{country}_{league}", str(int(season)))
        needed = [
            os.path.join(save_dir, "fixtures.json"),
            os.path.join(save_dir, "match_events.json"),
            os.path.join(save_dir, "players.json"),
        ]
        return all(os.path.exists(p) for p in needed)

    def _preflight_validate_json(self, country, league, season):
        """
        Cheap sanity checks before spawning heavy compute.
        Raises ValueError on malformed/empty structures.
        """
        import os, json

        save_dir = os.path.join(self.script_dir, f"{country}_{league}", str(int(season)))

        with open(os.path.join(save_dir, "fixtures.json"), "r", encoding="utf-8") as f:
            fixtures = json.load(f)
        with open(os.path.join(save_dir, "match_events.json"), "r", encoding="utf-8") as f:
            events = json.load(f)
        with open(os.path.join(save_dir, "players.json"), "r", encoding="utf-8") as f:
            players = json.load(f)

        if not isinstance(fixtures, dict) or "response" not in fixtures or not isinstance(fixtures["response"], list):
            raise ValueError("fixtures.json malformed (missing 'response' list)")
        if not isinstance(events, dict):
            raise ValueError("match_events.json malformed (expected dict keyed by fixture_id)")
        if not isinstance(players, dict):
            raise ValueError("players.json malformed (expected dict keyed by team_id)")

        # Optional: ensure at least one regular-season fixture looks present
        if len(fixtures["response"]) == 0:
            raise ValueError("fixtures.json has empty 'response'")

    def _run_single_season_subprocess(self, country, league, season) -> tuple[int, str]:
        """
        Run one SeasonAnalyzer job in a fresh Python process, returning (returncode, tail).
        Uses runpy to execute *this* script file and call SeasonAnalyzer directly.
        """
        import os, sys, subprocess, json

        # Pass job config via env to avoid shell-quoting pain
        job_env = {
            "country": country,
            "league": league,
            "season": int(season),
            "lasso_alphas": self.lasso_alphas,
            "minutes_equiv_tolerance": self.minutes_equiv_tolerance,
            "script_path": self.script_path,
        }
        env = os.environ.copy()
        env["SBATCH_JOB"] = json.dumps(job_env)

        # Keep BLAS/OpenMP single-threaded and DO NOT mask duplicate OMP
        env.pop("KMP_DUPLICATE_LIB_OK", None)
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("NUMEXPR_NUM_THREADS", "1")

        # One-liner: load this script via runpy, pull SeasonAnalyzer, construct, run
        code = r"""
import os, json, runpy
ns = runpy.run_path(os.environ["SBATCH_JOB_CONFIG_PATH"])
# If run_path returns None (some runners), try explicit path from env
if ns is None or "SeasonAnalyzer" not in ns:
    ns = runpy.run_path(os.environ["SCRIPT_PATH"])
SA = ns["SeasonAnalyzer"]
cfg = json.loads(os.environ["SBATCH_JOB"])
an = SA(
    league_name=cfg["league"],
    country_name=cfg["country"],
    season=int(cfg["season"]),
    lasso_alphas=cfg["lasso_alphas"],
)
# pass-through tolerance if attribute exists
if hasattr(an, "minutes_equiv_tolerance"):
    an.minutes_equiv_tolerance = int(cfg.get("minutes_equiv_tolerance", 0))
an.run()
"""
        # Provide both paths: some environments require SCRIPT_PATH only
        env["SCRIPT_PATH"] = self.script_path
        # Some distros require a separate config path var name; point to same script for safety
        env["SBATCH_JOB_CONFIG_PATH"] = self.script_path

        p = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, env=env)
        # Return code + short tail of stderr/stdout for logging
        tail = (p.stderr or p.stdout or "")[-2000:]
        return p.returncode, tail

    def run_all(self, limit: int | None = None) -> "pd.DataFrame":
        """
        Run SeasonAnalyzer for each completed row. Returns a batch report DataFrame.
        Each season runs in its own subprocess, so native crashes won't kill the batch.
        """
        import os, sys, time, json, traceback, pandas as pd

        rows = []
        df = self.progress_df.copy()
        # Keep only competitions that look complete
        df = df[df.apply(self._row_is_complete, axis=1)]

        if limit is not None:
            df = df.head(int(limit))

        for i, row in df.iterrows():
            country = str(row[self.col_country])
            league  = str(row[self.col_league])
            season  = int(row[self.col_season])

            started_at = time.time()
            status = "skipped_missing_inputs"
            message = ""
            outputs = {}

            try:
                if not self._json_inputs_exist(country, league, season):
                    status = "skipped_missing_inputs"
                    message = "JSON inputs not found (fixtures/events/players)."
                else:
                    # Preflight validation to “cast an error” for malformed datasets
                    try:
                        self._preflight_validate_json(country, league, season)
                    except Exception as ve:
                        status = "invalid_dataset"
                        message = f"Validation fail: {ve}"
                        raise RuntimeError("preflight_stop")

                    # Heavy compute in a fresh subprocess (survives native crashes)
                    rc, tail = self._run_single_season_subprocess(country, league, season)
                    if rc == 0:
                        status = "ok"
                        # We can optionally open the DB and count rows for this job (lightweight)
                        try:
                            import sqlite3
                            conn = sqlite3.connect(os.path.join(self.script_dir, "analysis_results.db"))
                            q = """
                                SELECT COUNT(1) AS n
                                FROM analysis_results
                                WHERE country=? AND league=? AND season=?
                            """
                            cdf = pd.read_sql_query(q, conn, params=[country, league, season])
                            conn.close()
                            outputs["rows_written"] = int(cdf["n"].iloc[0]) if not cdf.empty else None
                        except Exception:
                            # Non-fatal; keep going
                            pass
                    else:
                        status = "error_subprocess"
                        message = f"rc={rc}. tail:\n{tail}"

            except RuntimeError as rt:
                if str(rt) != "preflight_stop":
                    status = "error"
                    message = f"RuntimeError: {rt}"
            except Exception as e:
                status = "error"
                if self.debug:
                    message = "".join(traceback.format_exception(type(e), e, e.__traceback__))[:4000]
                    print(f"[ERROR] {country} / {league} / {season}\n{message}")
                else:
                    message = f"{type(e).__name__}: {e}"

            elapsed = round(time.time() - started_at, 2)
            rows.append({
                "country": country,
                "league": league,
                "season": season,
                "status": status,
                "message": message,
                "elapsed_s": elapsed,
                **outputs,
            })

        report = pd.DataFrame(rows)
        # Save batch report next to the progress file
        out_csv  = os.path.join(self.script_dir, "season_batch_report.csv")
        out_xlsx = os.path.join(self.script_dir, "season_batch_report.xlsx")
        report.to_csv(out_csv, index=False)
        report.to_excel(out_xlsx, index=False)
        print(f"[Batch] Saved: {out_csv}\n[Batch] Saved: {out_xlsx}")
        return report

 

if __name__ == "__main__":
    # Defaults: require all *_done flags; set debug=True for full traces
    runner = SeasonBatchRunner(
        progress_filename="downloads_progress.xlsx",
        filter_mode="all_done",       # or "events_only"
        minutes_equiv_tolerance=0,    # 0 for exact minutes; >0 for bucketing
        lasso_alphas=(0.01, 0.001),
        debug=True,                   # show full error tracebacks in console
    )
    runner.run_all()
    
    # jobs = [
    #     ("Germany", "Bundesliga"),
    #     ("Spain", "La Liga"),
    #     ("England", "Premier League"),
    # ]
    
    # # smallrunner = SeasonAnalyzer(league_name="Bundesliga", country_name="Germany", season=2022, lasso_alphas=[0.01, 0.001])
   
    # for year in range(2024, 2013, -1):
    #     for country, league in jobs:
            
    #         smallrunner = SeasonAnalyzer(league_name= league, country_name=country, season=year, lasso_alphas=[0.01, 0.001])
    #         smallrunner.run()
